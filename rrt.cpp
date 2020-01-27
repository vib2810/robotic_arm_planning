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
        bool obs=false;
        bool obs_check=false;

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
	double cost_h(info i1, info i2)
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
    Mat costmap;
    double dq=10;
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

    	tree::info start, end;
    	start.level=0;
    	for(int i=0; i<n_states; i++) 
		{
			start.x[i]=(theta_start[i]);
			end.x[i]=(theta_end[i]);
		}
    	this->tree_data= tree(&start);
    	
    	//Build Tree
    	for(int i=0; i<10000; i++) 
    	{
    		expand();
    		if(i%10 ==0 )
			{	
				//wrong logic, need to iterate through the whole tree and check connectivity;
				cout<<"Iterations: "<<i<<endl;
				tree::info *nearest_to_end=check_path(&end);
				if(nearest_to_end==NULL) cout<<"No path"<<endl;
				else 
				{
					cout<<"Path exists at i="<<i<<endl;
					waitKey(0);
				}
				visualize();
			}
    	}
    }
    tree::info *check_path(tree::info *end)
    {
		tree::info *nearest_to_end=tree_data.get_nearest(end);
		if(nearest_to_end==NULL) 
		{
			cout<<"case null"<<endl;
			return NULL;
		}
 		if(check_valid(nearest_to_end, end)==true) return nearest_to_end;
 		return NULL;
    }
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
 		
 		if(check_valid(nearest, sampled)==true)
 		{
 			tree_data.add_child(sampled, nearest);
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
    // circle(costmap, Point(img_res*(50-25)/100,img_res*(50-15)/100), img_res/30, Scalar(0), -1);
    // circle(costmap, Point(img_res*(50+32)/100,img_res*(50-14)/100), img_res/70, Scalar(0), -1);

    // circle(costmap, Point(500+220,500-30), 10, Scalar(0), -1);

    vector<double> start{180,180,0}, end{250,200,40};
    rrt plan(2);
    plan.plan_path(start, end, costmap);
}
