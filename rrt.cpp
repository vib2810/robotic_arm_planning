#include<stdio.h> 
#include<bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int img_res=1000;
double l[3]={ (double)250*img_res/(double)1000, (double)180*img_res/(double)1000.0, (double)80*img_res/(double)1000.0} ;//l0, l1, l2
double max_lim[3]={360, 360, 360};
double min_lim[3]={0, 0, 0};
// double reso[3]={5,5,5}; //resolution for each state
double reso[3]={10,18,36}; //resolution for each state


template <typename T> 
ostream& operator<<(ostream& os, const vector<T>& v) 
{ 
    os << "["; 
    for (int i = 0; i < v.size(); ++i) { 
        os << v[i]; 
        if (i != v.size() - 1) 
            os << ", "; 
    } 
    os << "]\n"; 
    return os; 
} 

class tree
{
public:
	struct info 
	{ 
		bool open=false;
        bool closed=false;
        bool end_connected=false;
        bool is_end=false;

        // bool obs_check=false;

        int n_states;
        double g,h,f;
        double temp_dist;

        vector<double> x; //position 
        vector<int> p; //parents
        
        info(int n_states_in=3) 
        { 
            n_states=n_states_in;
            x.assign(n_states, 0);
            p.assign(n_states, 0);
        }
        bool operator<(const info &b) const
        {
            return f>b.f;
        }  
	    vector<info> child; 
	    int level;
	}; 

	info *head;
	double dq=50;
	static double cost_h(info i1, info i2)
    {
        double temp=0;
        for(int i=0; i<i1.n_states; i++)
        {
            // temp+= pow(i1.x[i]-i2.x[i],2);
            temp+= min(pow(i1.x[i]-i2.x[i],2), pow(360-fabs(i1.x[i]-i2.x[i]),2));
        }
        return sqrt(temp);
    }

	void add_child(info *to_add, info *addr_to_add)
	{
		to_add->level=addr_to_add->level + 1;
		addr_to_add->child.push_back(*to_add);
	}

	info *get_nearest(info *to_add, info *head=NULL)
	{
		if(head==NULL) head=this->head;
		bool same=true;
		for(int i=0; i<to_add->x.size(); i++) 
		{
			if(to_add->x[i]!=head->x[i])
			{
				same=false;
			}
		}
		if(same==true) return NULL;
		info *min_add;
		info temp;
		double current_dist=cost_h(*head, *to_add);
		// double current_dist=cost_h(temp, to_add);

		head->temp_dist=current_dist;
		min_add=head;

		for(int i=0; i<head->child.size(); i++)
		{
			// cout<<"here1"<<endl;
			info* small_in_child=get_nearest(to_add, &(head->child[i]) );
			if(current_dist > small_in_child->temp_dist)
			{
				min_add=small_in_child;
				current_dist=small_in_child->temp_dist;
			}
			// current_dist=min(current_dist, small_in_child->temp_dist);
		}
		return min_add;
	}

	void print()
	{
		print(this->head);
	}
	void print(info *head)
	{
		for(int i=0; i<head->level; i++) cout<<"\t";
		cout<<head->x<<endl;
		for(int i=0; i<head->child.size(); i++)
		{
			print(&head->child[i]);
		}
	}
	Mat img_tree(info * head, Mat img)
	{
    	circle(img, Point(2*head->x[0], 2*head->x[1]), 2, Scalar(100,100,0), -1);
		for(int i=0; i<head->child.size(); i++)
		{
			line(img, Point(2*head->x[0], 2*head->x[1]), Point(2*head->child[i].x[0], 2*head->child[i].x[1]), Scalar(100,100,2), 1);
			img=img_tree(&head->child[i], img);
		}
    	return img;

	}
 	Mat visualize(Mat img)
 	{	
 		img=img_tree(this->head, img);
 		return img;
 	}

	tree(){ }
	tree(info *value)
	{
		head = value;
	}
};

class rrt
{
public:
 	friend tree;
 	tree tree_data;
    int n_states; //to represent no of variables in each state
    vector<vector<double>> path;
    vector<double>theta_start, theta_end;
    tree::info *end;
    Mat costmap;
    double dq=10;
    bool path_available=false;

 	rrt(int n_states)
 	{
 		srand (time(0));
    	this->n_states=n_states;
 	}

 	bool obs_check(vector<double> theta)
    {
        // draw(theta, 2, costmap);
        // cout<<"checking obstacle for "<<theta<<endl;
        int x[n_states], y[n_states];
        x[0]=l[0]*cos(theta[0]*CV_PI/180.0), y[0]=l[0]*sin(theta[0]*CV_PI/180.0);
        for(int i=1; i<n_states; i++)
        {
            x[i]=x[i-1]+l[i]*cos(theta[i]*CV_PI/180.0);
            y[i]=y[i-1]+l[i]*sin(theta[i]*CV_PI/180.0);
        }
        Mat temp(img_res, img_res, CV_8UC1,Scalar(255));   
        for(double l=0; l<=1; l+=0.1)
        {
            double l_x[n_states], l_y[n_states];
            
            //construct points to check;
            l_x[0]= l*0+(1-l)*x[0];
            l_y[0]= l*0+(1-l)*y[0];
            for(int i=1; i<n_states; i++)
            {
                l_x[i]= l*x[i-1]+(1-l)*x[i];
                l_y[i]= l*y[i-1]+(1-l)*y[i];
            }

            for(int i=0; i<n_states; i++)
            {
                if(isvalid(costmap, costmap.rows/2 + l_y[i], costmap.cols/2 +l_x[i])==1)
                { 
                    if(costmap.at<uchar>(costmap.rows/2 + (int)l_y[i],costmap.cols/2 +(int)l_x[i])==0)
                    {
                        // cout<<"Obstacle"<<endl;
                        return true;
                    }
                }
            }
        }
        // cout<<"No Obstacle"<<endl;
        // waitKey(0);
        return false;
    }
    int isvalid(tree::info input)
    {
        int flag=1;
        for(int i=0; i<input.n_states; i++)
        {
            if(!(min_lim[i]<=input.x[i] && input.x[i]<=max_lim[i])) 
            {
               flag=0;
               break;
            }
        }
        return flag;
    }
    int isvalid(Mat A, int i, int j)
    {
        if(i<A.rows && j< A.cols &&i>=0&&j>=0) return 1;
        return 0;
    }
    void plan_path(vector<double> theta_start, vector<double> theta_end, Mat costmap)
    {
    	this->costmap=costmap;
    	this->theta_start=theta_start;
    	this->theta_end=theta_end;

    	tree::info start;
    	start.level=0;
    	this->end = new tree::info;
    	// this->endis_end=true;
    	for(int i=0; i<n_states; i++) 
		{
			start.x[i]=(theta_start[i]);
			end->x[i]=(theta_end[i]);
		}
    	this->tree_data= tree(&start);
    	
    	//Build Tree
    	for(int i=0; i<10000; i++) 
    	{
    		expand();
    		// tree_data.print();
			// waitKey(0);

    		if(i%100 ==0 )
			{	
				cout<<"Iterations: "<<i<<endl;
				if(this->path_available==false) cout<<"No path"<<endl;
				else 
				{
					cout<<"Path exists at i="<<i<<endl;
				}
				visualize();
				waitKey(0);
			}
    	}
    }
    // void find_path_astar()
    // {
    //     cout<< "-------------------------------finding path in tree----------------------------"<<endl;
    //     priority_queue<tree::info*> open; //open list
    //     open.push(tree_data->head); //Initialize open list by adding starting node 
    //     tree_data->head->open=true;
        
    //     int flag=0; //turns 1 if astar reaches destination;
    //     int count=0;
    //     waitKey(0);

    //     while(open.empty()!=1 && flag!=1 && count<1000000)   
    //     {
    //         ++count;
    //         info q=open.top();
    //         open.pop();
            
    //         // arr[get_index_from_value(q.x[0],0)][get_index_from_value(q.x[1],1)].open = false; //remove q from open list
    //     	q->open=false;
    //         // if(arr[get_index_from_value(q.x[0],0)][get_index_from_value(q.x[1],1)].closed == true ) continue;
    //         if(q->closed == true ) continue;

    //         if(count%100==0) cout<<"Iteration count: " <<count << " States: "<<q.x[0]<<" "<<q.x[1]<<" Cost: "<<q.g<<"+"<<q.h<<" ="<<q.f<<endl;
          
    //       	for(int i=0; i<q->child.size(); q++)
    //       	{
    //       		tree::info *temp=new tree::info;
    //       		temp->open=true;          //Set the pi,pj,h,g,f values to the variable temp
    //             for(int j=0; j<n_states; j++)
    //             {
    //             	temp->x[j]=q->child.x[j];
    //             	temp->p[j]=q->x[j];
    //             }
    //             temp->g= q.g+tree::cost_h(*q->child[i], *q);
    //             temp->h= tree::cost_h(*q->child[i],*this->end);
    //             temp->f= temp->g + temp->h ;
               
    //             if(q->child[i].is_end==true) //check if its destination
    //             {
    //             	for(int j=0; j<n_states; j++) q->child[i].p[j]=temp->p[j];
    //                 flag=1; //reached destination
    //                 break;
    //             }
    //             // if()	
    //         //     if( arr[ get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].open==true && arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].g < temp.g) continue; //If node is present in open list with lower f skip this node
    //         //     if( arr[ get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].closed==true && arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].g < temp.g) continue; //If node is present in closed list with lower f skip this node
    //         //     float fac1=0.8/f_fac, fac2=1/f_fac, fac3=1.1/f_fac;
    //       		// tree::info *temp=new tree::info;

    //       	}
    //     }
    //         // for(int i=-1;i<2;i++)//Traverse through  //q is parent node, temp is the successor
    //         // {
    //         //     for(int j=-1;j<2;j++)
    //         //     {
    //         //         info temp(n_states);
    //         //         temp.x[0]=add_ang(q.x[0],i*reso[0]);
    //         //         temp.x[1]=add_ang(q.x[1],j*reso[1]);
                            
    //         //         // double state={temp.x[0], temp.x[1]};

    //         //         if(isvalid(temp)==1 && (i!=0 || j!=0) && arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].obs==false)  //valid neighbour + not the central +not an obstacle nearby check- check
    //         //         {
    //         //             if(i==0 && j==0) continue;
    //         //             int index[n_states]={get_index_from_value(temp.x[0],0),get_index_from_value(temp.x[1],1)};
    //         //             if(arr[index[0]][index[1]].obs_check==false)
    //         //             {
    //         //                 arr[index[0]][index[1]].obs=obs_check(arr[index[0]][index[1]].x, n_states, costmap);
    //         //                 arr[index[0]][index[1]].obs_check=true;
    //         //             }
    //         //             if(arr[index[0]][index[1]].obs==true) continue;
    //         //             temp.open=true;          //Set the pi,pj,h,g,f values to the variable temp
    //         //             temp.p[0]=get_index_from_value(q.x[0],0);
    //         //             temp.p[1]=get_index_from_value(q.x[1],1);
    //         //             temp.g= q.g+cost2[i+1][j+1];
    //         //             temp.h= ch(temp,end);
    //         //             temp.f= temp.g + temp.h ;
                       
    //         //             if( temp.x[0]==end.x[0] && temp.x[1]==end.x[1]) //check if its destination
    //         //             {
    //         //                 arr[get_index_from_value(temp.x[0],0)][ get_index_from_value(temp.x[1],1)].p[0] = temp.p[0];  //Save parent details for destination
    //         //                 arr[get_index_from_value(temp.x[0],0)][ get_index_from_value(temp.x[1],1)].p[1] = temp.p[1];
    //         //                 flag=1; //reached destination
    //         //                 break;
    //         //             }

    //         //             if( arr[ get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].open==true && arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].g < temp.g) continue; //If node is present in open list with lower f skip this node
    //         //             if( arr[ get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].closed==true && arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)].g < temp.g) continue; //If node is present in closed list with lower f skip this node
    //         //             float fac1=0.8/f_fac, fac2=1/f_fac, fac3=1.1/f_fac;
    //         //             // int f1=fac*temp.h, f2=fac*temp.h*temp.h;
    //         //             circle(config_space, Point(get_index_from_value(temp.x[0], 0),get_index_from_value(temp.x[1], 1) ),1, Scalar(temp.h*fac1,temp.h*fac2,temp.h*fac3), -1, 4);
    //         //             //Otherwise add node to open list
    //         //             open.push(temp); //Push to priority list
    //         //             arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)] = temp;
    //         //         }
    //         //     }
    //         // }
    //         // arr[get_index_from_value(q.x[0],0)][get_index_from_value(q.x[1],1)].closed=true;
    //     }
    //     cout<<"Done Plan"<<endl;
    //     if(flag==1)
    //     {
    //         cout<<"Path Found"<<endl;
    //         info temp=arr[get_index_from_value(end.x[0],0)][get_index_from_value(end.x[1],1)];
    //         while(temp.x[0]!=start.x[0]||temp.x[1]!=start.x[1])
    //         {
    //             {
    //                 vector<double> temp_arr;
    //                 temp_arr.push_back(temp.x[0]);
    //                 temp_arr.push_back(temp.x[1]);
    //                 config_space.at<Vec3b>(get_index_from_value(temp.x[1], 1),get_index_from_value(temp.x[0], 0))[0]=0;
    //                 config_space.at<Vec3b>(get_index_from_value(temp.x[1], 1),get_index_from_value(temp.x[0], 0))[1]=255;
    //                 config_space.at<Vec3b>(get_index_from_value(temp.x[1], 1),get_index_from_value(temp.x[0], 0))[2]=0;

    //                 // circle(config_space, Point(get_index_from_value(temp.x[0], 0),get_index_from_value(temp.x[1], 1) ),1, Scalar(0,200,200), -1, 4);
    //                 path.push_back(temp_arr);
    //                 temp=arr[temp.p[0]][temp.p[1]];
    //             }
    //         }
    //         imshow("Configuration Space", config_space);
    //         return;
    //     }
    //     else 
    //     {
    //         cout<<"No path"<<endl;
    //         return;
    //     }

    // }
 	bool expand()
 	{
 		tree::info *sampled= new tree::info;
 		for(int i=0; i<n_states; i++)
 		{
 			sampled->x[i]= static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/max_lim[i]));
 		}
		tree::info *nearest=tree_data.get_nearest(sampled);
		if(nearest==NULL) return false;
		
		double dist=0;
		for(int i=0; i<n_states; i++) dist+=pow(sampled->x[i]-nearest->x[i],2);
		dist=sqrt(dist);
 		if(dist>this->dq) for(int i=0; i<n_states; i++) sampled->x[i]= nearest->x[i]+((sampled->x[i]-nearest->x[i])*this->dq)/dist;
 		sampled->h=tree::cost_h(*sampled, *this->end);
 		if(check_valid(nearest, sampled)==true)
 		{
 			// bool flag=false;
 			if(check_valid(sampled, this->end)==true)
 			{
 				this->path_available=true;
 				sampled->end_connected=true;
				// tree_data.add_child(this->end, sampled);
 				// flag = true;
 			}
 			else sampled->end_connected=false;

 			tree_data.add_child(sampled, nearest);
 		// 	if(flag==true) 
			// {
			// 	cout<<"adding end"<<endl;
			// }
 			return true;
 		}
 		return false;
 	}

 	bool check_valid(tree::info * nearest, tree::info * sampled)
 	{
 		double x[n_states], y[n_states];
		for(double l=0; l<=1; l+=0.1)
		{	
			vector<double> theta;
			for(int i=0; i<n_states; i++)
	        {
	        	theta.push_back(l*nearest->x[i]+(1-l)*sampled->x[i]);
	        }
	        if(obs_check(theta)==1) return false;
	    }
	    return true;
 	}

 	void visualize()
 	{
		Mat viz(2*max_lim[0],2*max_lim[1], CV_8UC3, Scalar(255,255,255));  
    	circle(viz, Point(2*this->theta_start[0],2*this->theta_start[1]), 20, Scalar(0,255,0), -1);
    	circle(viz, Point(2*this->theta_end[0],2*this->theta_end[1]), 20, Scalar(0,0,255), -1);

 		tree_data.visualize(viz);	
 		namedWindow("viz", WINDOW_NORMAL);
    	imshow("viz", viz);
    	waitKey(10);
 	}
};

int main() 
{ 
	Mat costmap(img_res, img_res, CV_8UC1,Scalar(255));   
    circle(costmap, Point(img_res*(50+10)/100,img_res*(50+10)/100), img_res/30, Scalar(0), -1);
    circle(costmap, Point(img_res*(50-25)/100,img_res*(50-15)/100), img_res/30, Scalar(0), -1);
    // circle(costmap, Point(img_res*(50+32)/100,img_res*(50-14)/100), img_res/70, Scalar(0), -1);

    // circle(costmap, Point(500+220,500-30), 10, Scalar(0), -1);

    vector<double> start{180,180,0}, end{250,200,40};
    rrt plan(2);
    plan.plan_path(start, end, costmap);
}
