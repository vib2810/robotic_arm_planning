#include <opencv2/opencv.hpp>
#include <iostream>
#include <bits/stdc++.h>
#include <limits.h>
#include <math.h>
using namespace cv;
using namespace std;

int img_res=1000;
double l[3]={ (double)250*img_res/(double)1000, (double)180*img_res/(double)1000.0, (double)80*img_res/(double)1000.0} ;//l0, l1, l2
double max_lim[3]={360, 360, 360};
double min_lim[3]={0, 0, 0};
// double reso[3]={5,5,5}; //resolution for each state
double reso[3]={10,18,36}; //resolution for each state
// double reso[3]={30,30,30}; //resolution for each state

void drawstate(vector<double>theta, int n_states, Mat &image, Scalar color)
{
    int x_off=image.rows/2, y_off=image.cols/2;
    int x[n_states], y[n_states];
    x[0]=l[0]*cos(theta[0]*CV_PI/180.0), y[0]=l[0]*sin(theta[0]*CV_PI/180.0);
    for(int i=1; i<n_states; i++)
    {
        x[i]=x[i-1]+l[i]*cos(theta[i]*CV_PI/180.0);
        y[i]=y[i-1]+l[i]*sin(theta[i]*CV_PI/180.0);
    }
    line(image, Point(x_off,y_off), Point(x_off+x[0],y_off+y[0]), color, img_res/100);
    
    for(int i=1; i<n_states; i++)  line(image, Point(x_off+x[i],y_off+y[i]), Point(x_off+x[i-1],y_off+y[i-1]), color, img_res/100);
    for(int i=0; i<n_states; i++)  circle(image, Point(x_off+x[i],y_off+y[i]), img_res/90, Scalar(255,0,0), -1);
    
}
void draw(vector<double> theta, int n_states, Mat costmap) //input in degrees, n_states=3
{
    int x_off=costmap.rows/2, y_off=costmap.cols/2;
    Mat temp( costmap.rows, costmap.cols ,CV_8UC3,Scalar(255,255,255));   

    for(int i=0; i<temp.rows; i++)
    {
        for(int j=0; j<temp.cols; j++)
        {
            if(costmap.at<uchar>(i,j)==0)
            {
                temp.at<Vec3b>(i,j)[0]=0;
                temp.at<Vec3b>(i,j)[1]=0;
                temp.at<Vec3b>(i,j)[2]=0;
            }
        }
    }
    line(temp, Point(x_off*((100-10)/(double)100),y_off), Point(x_off*((100+10)/(double)100),y_off), Scalar(255,122,122),  img_res/80); //Base line
    circle(temp, Point(x_off,y_off),  img_res/90, Scalar(255,0,0), -1); //Base circle

    drawstate(theta, n_states, temp, Scalar(0,0,255));
    namedWindow("Map", WINDOW_NORMAL);
    imshow("Map",temp);
    // waitKey(300000);
    return;
}
void draw(vector<double> theta, int n_states, Mat costmap, vector<double> start, vector<double> end) //input in degrees, n_states=3
{
    int x_off=costmap.rows/2, y_off=costmap.cols/2;
    Mat temp( costmap.rows, costmap.cols ,CV_8UC3,Scalar(255,255,255));   

    for(int i=0; i<temp.rows; i++)
    {
        for(int j=0; j<temp.cols; j++)
        {
            if(costmap.at<uchar>(i,j)==0)
            {
                temp.at<Vec3b>(i,j)[0]=0;
                temp.at<Vec3b>(i,j)[1]=0;
                temp.at<Vec3b>(i,j)[2]=0;
            }
        }
    }
    line(temp, Point(x_off*((100-10)/(double)100),y_off), Point(x_off*((100+10)/(double)100),y_off), Scalar(255,122,122),  img_res/80); //Base line
    circle(temp, Point(x_off,y_off),  img_res/90, Scalar(255,0,0), -1); //Base circle

    drawstate(theta, n_states, temp, Scalar(0,0,255));
    drawstate(end, n_states, temp, Scalar(0,10,100));
    drawstate(start, n_states, temp, Scalar(10,100,0));

    namedWindow("Map", WINDOW_NORMAL);
    imshow("Map",temp);
    waitKey(10);
    return;
}
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


class astar
{
public: 
    int n_states; //to represent no of variables in each state
    Mat costmap;
    vector<vector<double>> path;
    vector<int> s;
    double cost2[3][3]={{ 1.414, 1, 1.414},
                       {     1, 0,     1},
                       { 1.414, 1, 1.414}};
    double cost3[3][3][3]=  //realcost(a,b,c) = cost[a+i][b+j][c+k] ,i,j,k got from -1 to 1
    {
            { //a=-1
              {1.7,1.4, 1.7}, //b=-1
              {1.4,  1, 1.4}, //b= 0
              {1.7,1.4, 1.7}  //b= 1
            },
            { //a=0
              {1.4, 1, 1.4},  
              {  1, 0,   1},
              {1.4, 1, 1.4}
            },
            { //a=1
              {1.7,1.4, 1.7}, //b=-1
              {1.4,  1, 1.4}, //b= 0
              {1.7,1.4, 1.7}  //b= 1
            },
    };
    typedef struct info
    {
        bool open=false;
        bool closed=false;
        bool obs=false;
        bool obs_check=false;

        int n_states;
        double g,h,f;
        
        vector<double> x; //position 
        vector<int> p; //parents
        
        info(int n_states_in=3) 
        { 
            n_states=n_states_in;
            x.assign(n_states, 0);
            p.assign(n_states, 0);

            // x=new double[n_states];
            // p=new int[n_states];
        }
        bool operator<(const info &b) const
        {
            return f>b.f;
        }  
    } info;

    int get_index_from_value(double x_i, int i)
    {
        int temp=(int)((x_i-min_lim[i])/(float)reso[i]);
        // cout<<temp<<" "<<(int)((max_lim[i]-min_lim[i])/(float)reso[i])<<endl;
        if(temp>=(int)((max_lim[i]-min_lim[i])/(float)reso[i])) temp=(int)((max_lim[i]-min_lim[i])/(float)reso[i])-1;
        return temp;
    }

    int isvalid(info input)
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
    float f_fac=0.05;
    double ch(info i1, info i2)
    {
        double temp=0;
        for(int i=0; i<i1.n_states; i++)
        {
            temp+= min(pow(i1.x[i]-i2.x[i],2), pow(360-fabs(i1.x[i]-i2.x[i]),2));
        }
        return f_fac*sqrt(temp);
        // return 0;
    }


    astar(int n_states)
    {
        this->n_states=n_states;
        this->s.assign(n_states, 0);
        for(int i=0; i<this->n_states; i++) this->s[i]=(max_lim[i]-min_lim[i])/reso[i];
    }
    double add_ang(double a, double b)
    {
        double temp=a+b;
        if(temp<0) temp=temp+360;
        if(temp>=360) temp=360-temp;

        return temp;
    }
    bool obs_check(vector<double> theta, int n_states, Mat costmap)
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
    int clamp(int input, int reso)
    {
        if(input%reso!=0)
        {
            if(input%reso > (reso-(input%reso))) input=input+reso-(input%reso);
            else input=input-input%reso;
            return input;
        }
        else return input;
    }
    void plan_path(vector<double> theta_start, vector<double> theta_end, int n_states, Mat costmap)
    {
        this->costmap=costmap;
        // make 2d array of info
        info*** arr = new info**[s[0]]; 
        for(int i = 0; i < s[0]; ++i)  
        {
            arr[i] = new info*[s[1]];
            for(int j=0; j<s[1]; ++j) arr[i][j] = new info[s[2]];
        }

        // Mat config_space(s[0], s[1], CV_8UC3,Scalar(255));   
        cout<<"done allocation"<<endl;
        // namedWindow("Configuration Space", WINDOW_NORMAL);
        //initialize array
        for(int i=0;i<s[0];i++) 
        {
            for(int j=0;j<s[1];j++)
            {
                for(int k=0; k<s[2]; k++)
                {
                    arr[i][j][k].f=FLT_MAX;
                    arr[i][j][k].x[0]=min_lim[0]+i*reso[0];
                    arr[i][j][k].x[1]=min_lim[1]+j*reso[1];
                    arr[i][j][k].x[2]=min_lim[2]+k*reso[2];
                    // arr[i][j][k].obs=obs_check(arr[i][j][k].x, n_states, costmap);    
                    // arr[i][j][k].obs_check=true;                    

                    // circle(config_space, Point(i,j), 1, Scalar(arr[i][j].obs*105,arr[i][j].obs*5,arr[i][j].obs*5), -1);
                    arr[i][j][k].open=false;
                }
            }
        }  

        //define start and end nodes
        info start(n_states),end(n_states);
        for(int i=0; i<n_states; i++)
        {
            theta_end[i]=clamp(theta_end[i], reso[i]);
            theta_start[i]=clamp(theta_start[i], reso[i]);
            end.x[i]=theta_end[i];
            start.x[i]=theta_start[i];
        }

        start.open=true;
        start.g=0;
        start.h=ch(start, end);
        start.f=start.g+start.h;

        cout<<"Start: "<< start.x <<endl;
        cout<<"End: "<< end.x <<endl;
        cout<<"Configuration Space Size: "<<s<<endl;
        draw(theta_start, 3, costmap, theta_start, theta_end);
        // circle(config_space, Point(get_index_from_value(start.x[0], 0),get_index_from_value(start.x[1], 1) ), 1, Scalar(0,255,0), -1);
        // circle(config_space, Point(get_index_from_value(end.x[0], 0),get_index_from_value(end.x[1], 1) ),1, Scalar(0,0,255), -1);


        if(arr[get_index_from_value(start.x[0], 0)][ get_index_from_value(start.x[1], 1)][ get_index_from_value(start.x[2], 2)].obs==true || arr[get_index_from_value(end.x[0], 0)][ get_index_from_value(end.x[1], 1)][get_index_from_value(end.x[2], 2)].obs==true )
        {
            cout<<"Cant be planned, init and final states lie on the plan"<<endl;
            return;
        }
        priority_queue<info> open; //open list
        open.push(start); //Initialize open list by adding starting node 
        arr[ get_index_from_value(start.x[0], 0)][ get_index_from_value(start.x[1], 1)][ get_index_from_value(start.x[2], 2)].open=true;
        
        // imshow("Configuration Space", config_space);
        // waitKey(1);
        cout<< "-------------------------------done setup----------------------------"<<endl;
        int flag=0; //turns 1 if astar reaches destination;
        int count=0, count1=0;
        waitKey(0);
        cout<<"Started Planning"<<endl;
        while(open.empty()!=1 && flag!=1 && count<100000000)   
        {
            info q=open.top();
            open.pop();
            arr[get_index_from_value(q.x[0],0)][get_index_from_value(q.x[1],1)][get_index_from_value(q.x[2],2)].open = false; //remove q from open list
            if(arr[get_index_from_value(q.x[0],0)][get_index_from_value(q.x[1],1)][get_index_from_value(q.x[2],2)].closed == true ) continue;
            ++count, ++count1;

            
            // circle(config_space, Point(get_index_from_value(q.x[0], 0),get_index_from_value(q.x[1], 1) ),1, Scalar(255,100,100), -1);
            if(count1>500)
            {
                cout<<"Iteration count: " <<count << " States: "<<q.x<<" Cost: "<<q.g<<"+"<<q.h<<" ="<<q.f<<endl;
                draw(q.x, 3, costmap, theta_start, theta_end);
                // circle(config_space, Point(get_index_from_value(q.x[0], 0),get_index_from_value(q.x[1], 1) ),1,Scalar(0,0,255), -1);
                // imshow("Configuration Space", config_space);
                // waitKey(1);
                
                // circle(config_space, Point(get_index_from_value(q.x[0], 0),get_index_from_value(q.x[1], 1) ),1,Scalar(255,0,255), -1);
                // imshow("Configuration Space", config_space);

                // waitKey(1);
                count1=0;
            }
            for(int i=-1;i<2;i++)//Traverse through  //q is parent node, temp is the successor
            {
                for(int j=-1;j<2;j++)
                {
                    for(int k=-1;k<2;k++)
                    {
                        info temp(n_states);
                        temp.x[0]=add_ang(q.x[0],i*reso[0]);
                        temp.x[1]=add_ang(q.x[1],j*reso[1]);
                        temp.x[2]=add_ang(q.x[2],k*reso[2]);

                        if(isvalid(temp)==1 && (i!=0 || j!=0 || k!=0))  //valid neighbour + not the central +not an obstacle nearby check- check
                        {
                            if(i==0 && j==0 && k==0) continue;
                            int index[n_states]={get_index_from_value(temp.x[0],0),get_index_from_value(temp.x[1],1),get_index_from_value(temp.x[2],2)};
                            if(arr[index[0]][index[1]][index[2]].obs_check==false)
                            {
                                arr[index[0]][index[1]][index[2]].obs=obs_check(arr[index[0]][index[1]][index[2]].x, n_states, costmap);
                                arr[index[0]][index[1]][index[2]].obs_check=true;
                            }
                            if(arr[index[0]][index[1]][index[2]].obs==true) continue;
                            
                            temp.open=true;          //Set the pi,pj,h,g,f values to the variable temp
                            temp.p[0]=get_index_from_value(q.x[0],0);
                            temp.p[1]=get_index_from_value(q.x[1],1);
                            temp.p[2]=get_index_from_value(q.x[2],2);

                            temp.g= q.g+cost3[i+1][j+1][k+1];
                            temp.h= ch(temp,end);
                            temp.f= temp.g + temp.h ;
                           
                            if( temp.x[0]==end.x[0] && temp.x[1]==end.x[1] && temp.x[2]==end.x[2]) //check if its destination
                            {
                                arr[get_index_from_value(temp.x[0],0)][ get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)].p[0] = temp.p[0];  //Save parent details for destination
                                arr[get_index_from_value(temp.x[0],0)][ get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)].p[1] = temp.p[1];
                                arr[get_index_from_value(temp.x[0],0)][ get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)].p[2] = temp.p[2];

                                flag=1; //reached destination
                                break;
                            }

                            if( arr[ get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)].open==true 
                            && arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)].f < temp.f) continue; //If node is present in open list with lower f skip this node
                            
                            if( arr[ get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)].closed==true 
                            && arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)].f < temp.f) continue; //If node is present in closed list with lower f skip this node
                            
                            // float fac1=2/f_fac, fac2=3/f_fac, fac3=4/f_fac;
                            // int f1=fac*temp.h, f2=fac*temp.h*temp.h;
                            // circle(config_space, Point(get_index_from_value(temp.x[0], 0),get_index_from_value(temp.x[1], 1) ),1, Scalar(temp.h*fac1,temp.h*fac2,temp.h*fac3), -1, 4);
                            //Otherwise add node to open list
                            open.push(temp); //Push to priority list
                            arr[get_index_from_value(temp.x[0],0)][get_index_from_value(temp.x[1],1)][get_index_from_value(temp.x[2],2)] = temp;
                        }
                    }
                }
            }
            arr[get_index_from_value(q.x[0],0)][get_index_from_value(q.x[1],1)][get_index_from_value(q.x[2],2)].closed=true;
        }
        cout<<"Done Plan in Iterations: "<<count<<endl;
        if(flag==1)
        {
            cout<<"Path Found"<<endl;
            info temp=arr[get_index_from_value(end.x[0],0)][get_index_from_value(end.x[1],1)][get_index_from_value(end.x[2],2)];
            while(temp.x[0]!=start.x[0]||temp.x[1]!=start.x[1]||temp.x[2]!=start.x[2])
            {
                {
                    path.push_back(temp.x);
                    temp=arr[temp.p[0]][temp.p[1]][temp.p[2]];
                }
            }
            path.push_back(temp.x);
            return;
        }
        else 
        {
            cout<<"No path"<<endl;
            return;
        }
    }
    void draw_path()
    {
        waitKey(0);
        if(path.size()!=0)
        {
            cout<<path.size()<<endl;
            vector<double> end=path[0];
            vector<double> start=path[path.size()-1];
            for(int i=0; i<path.size(); i++)
            {
                vector<double> current=path[path.size()-1-i];
                if(i==path.size()-1) 
                {
                    draw(current, 3, costmap, start, end);
                    return;
                }
                vector<double> next=path[path.size()-2-i];
                cout<<"------------current: "<<current<<endl;
                cout<<"------------Next: "<<next<<endl;
                for(int i=0; i<n_states; i++)
                {
                    if(next[i]-current[i]>180) current[i]+=360;
                    if(current[i]-next[i]>180) next[i]+=360;

                }
                for(float j=0; j<1; j+=0.2)
                {
                    vector<double> inter;
                    for(int k=0; k<n_states; k++) 
                    {
                        inter.push_back(  (1-j)*current[k]+j*next[k] ) ;
                    }
                    cout<<"Drawing: "<<inter<<endl;
                    draw(inter, 3, costmap, start, end);
                    waitKey(60);
                }
                waitKey(60);
           }
       }
    }
};

int main(int argc,char ** argv)
{  
    Mat costmap(img_res, img_res, CV_8UC1,Scalar(255));   
    circle(costmap, Point(img_res*(50+10)/100,img_res*(50+10)/100), img_res/30, Scalar(0), -1);
    circle(costmap, Point(img_res*(50-25)/100,img_res*(50-15)/100), img_res/30, Scalar(0), -1);
    // circle(costmap, Point(img_res*(50+32)/100,img_res*(50-14)/100), img_res/70, Scalar(0), -1);

    // circle(costmap, Point(500+220,500-30), 10, Scalar(0), -1);

    vector<double> start{0,50,0}, end{160,200,40};
    astar plan(3);
    plan.plan_path(start, end, 3, costmap);
    while (1) plan.draw_path();
    // plan.astar1();
}

    // for(int i=0; i<=180; i=i+=20)
    // {
    //     for (int j=0;j<=180; j+=20)
    //     {
    //             // cout<<i<<" "<<j<<" "<<k<<endl;
    //             double theta[2]={i, j};
    //             drawstate(theta, 2);   
    //     }
    // } 

  // //3D Array of info 
        // int s1=180/reso[0], s2=180/reso[1], s3=180/reso[2];
        // info*** arr = new info **[s1];
        // for (int i=0; i<=s1; i++)
        // {
        //     arr[i] = new info *[s2];
        //     for (int j = 0; j < s2; j++)  arr[i][j] = new info [s3];
        // }
