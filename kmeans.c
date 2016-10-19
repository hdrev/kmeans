//
// Created by sestens on 9/13/16.
// Santiago Estens
// R10825470
// K-means clustering algorithm
// partially based  on the paper by MacQueen, J. "Some Methods for Classification and Analysis of Multivariate Observations."


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>


#define N 10000
#define threshold pow(10,-6)


int kmeans(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size);
double euclidean_distance_8dims(double *point1, double *point2);
int find_closest_cluster(int k,int dim, double *cluster_center,double *data);
int choose_initial_centers(int dim, double *data, int k, double *cluster_center);
int find_farthest_point(int dim, double *data, double *point);
int kmeans2(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size);
double sum_of_squared_errors(int dim, int k, double *cluster_center, int *cluster_size, double *test);
int max_min_newcenter(int dim, double *data, int k, double *cluster_center,int in);
int bisecting_kmeans(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size);
void squared_standard_deviation(int dim, int k, double *cluster_center, int *cluster_size, double *test,double *cluster_ssd);
int kmeans3(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size, double *test, int max_clusterssd,int count_k);
int max_index(double *array,int array_size);
int farthest_point_withinc(int dim, double *test, double *point,int max_clusterssd, int *cluster_size);
int kmeans2_2(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size);
int main(){
    int k, dim;
    //k can be 4,8,16,32
    k=4;
    dim=8;
    //cluster_asssign is an array of size N
    //where cluster_assign[i] indicates which of the k clusters the i-th datum is assigned to.
    int cluster_assign[N];
    //cluster_center[] is an array of size dim*k
    //where cluster_center[0] - cluster_center[dim-1] indicate the location of the center of the 0th cluster.
    double cluster_center[dim*k];
    //cluster_size[] is an array of size k
    //where cluster_size[i] indicates how many data items the i-th cluster has
    int cluster_size[k];
    //data[] is an array of size dim*N
    double data[dim*N];
    time_t t;
    srand((unsigned) time(&t));
    //Generate random data and store it in data
    for(int i=0;i<N*dim;i++){
        data[i]=(double)rand()/RAND_MAX;
    }

//    test euclidean distance function
//    ---------------------------------
//    printf("printing the points that are going to be passed to the euclidean distance\n");
//    for(int i=0;i<17;i++){
//        printf("%dn:-%lf\n",i,A[i]);
//
//    }
//    printf("distance=%lf",euclidean_distance_8dims(A,&A[8]));

//    //test find closest cluster
//    -------------------------------
//    //generate data for cluster center
//    for (int j = 0; j < k * dim; ++j) {
//        cluster_center[j]=rand();
//    }
//    printf("testing the closest cluster function\n");
//    printf("the point I'm passing is:\n");
//    //loop to print point
//    for (int l = 0; l < dim; ++l) {
//        printf("%d:--%lf\n",l,data[l]);
//    }
//    printf("the cluster points are:\n");
//    //loop to display cluster centers
//    for (int m = 0; m < k * dim; ++m) {
//        printf("%d:--%lf\n",m,cluster_center[m]);
//    }
//    printf("the closest cluster to the first datum is:%d\n",find_closest_cluster(k,dim,cluster_center,data)/8);

    //test kmeans
//  ------------------------
    //results: the clusters are having more points than the input N
    // and for some reason dim and k are being modified during kmeans execution
//    kmeans(dim,data,k,cluster_assign,cluster_center,cluster_size);
//    k=4;
//    dim=8;
//    printf("Printing results of the K means clustering algorithm\n");
//    for (int j = 0; j < k; ++j) {
//        printf("Cluster %d contains %d points.\n",j,cluster_size[j]/2);
//    }
//    printf("The cluster centers are:\n");
//    for (int l = 0; l < k*dim; ++l) {
//        printf("%d:--%lf\n",l,cluster_center[l]);
//    }
    //test find farthest point
//  ----------------------------
    //create a random point to simulate cluster center
//    double tpoint[dim];
//    int tdist;
//    for (int j = 0; j < dim; ++j) {
//        tpoint[j]=rand();
//    }
//    tdist=find_farthest_point(dim,data,tpoint);
//    printf("The farthest point index from data to tpoint is:= %d\nThe distance was := %lf",tdist,euclidean_distance_8dims(tpoint,&data[tdist]));
    //test kmeans2
//  --------------
//    kmeans2(dim,data,k,cluster_assign,cluster_center,cluster_size);
//    printf("Printing results of the K means clustering algorithm\n");
//    for (int j = 0; j < k; ++j) {
//        printf("Cluster %d contains %d points.\n",j,cluster_size[j]);
//    }
//    printf("The cluster centers are:\n");
//    for (int l = 0; l < k*dim; ++l) {
//        printf("%d:--%lf\n",l,cluster_center[l]);
//    }
    //-----------------------
    //going to test bisecting kmeans
    bisecting_kmeans(dim,data,k,cluster_assign,cluster_center,cluster_size);
    printf("Printing results of the K means clustering algorithm\n");
    for (int j = 0; j < k; ++j) {
        printf("Cluster %d contains %d points.\n",j,cluster_size[j]);
    }
    printf("The cluster centers are:\n");
    for (int l = 0; l < k*dim; ++l) {
        printf("%d:--%lf\n",l,cluster_center[l]);
    }
    return 0;
}

double euclidean_distance_8dims(double *point1, double *point2){
    double distance=0;
    for(int i=0;i<8;i++){
        distance+=pow((*point1-*point2),2);
        point1++;
        point2++;
    }
    distance = sqrt(distance);
    return distance;
}

int kmeans(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size){
    int index;
    double change;
    int loop_counter=0;
    double cluster_points[k][N*dim];
    for (int m = 0; m < k; ++m) {
        for (int i = 0; i < N * dim; ++i) {
            cluster_points[k][i] = 0;
        }
    }
    double array[dim*k];
    //initialize array to 0
    memset(array, 0, dim*k*sizeof(double) );
    //start with k random initial centers
    for(int i=0;i<k*dim;i++){
        cluster_center[i]=rand();
    }
    //initialize cluster_assign, using -1 to denote that the points have no cluster yet
    for(int i=0;i<N;i++){
        cluster_assign[i]=-1;
    }
    //initialize cluster size initially to 0
    for (int i = 0; i < k; i++) {
        cluster_size[i]=0;
    }
    //each data point calculates its distance to the k centers.
    //assign the data point to the cluster whose center is the closest
    do{
        change=0;
        for (int g = 0; g < N * dim; g+=dim) {
            //find the array index of the closest cluster center
            index=find_closest_cluster(k,dim,cluster_center,&data[g]);
            //if index/dim != cluster_assign[g] the point has a new cluster,
            //increment change variable
            if(index/dim != cluster_assign[g/dim]){
                change++;
            }
            //assign membership of point
            cluster_assign[g/dim]=index/dim;
            //update size of the cluster
            cluster_size[index/dim]++;
            //update cluster_points
            for (int i = 0; i < dim; ++i) {
                cluster_points[index/dim][g+i] = data[g+i];
            }
            //update cluster centroid
            //by first summing the points in an array ------then get the mean and use that to update cluster_center of the clusters
            for (int j = 0; j < cluster_size[index/dim]*dim; ++j) {
                array[(index+j)%dim] += cluster_points[index/dim][j];
            }

        }
        //use a for loop to get the mean and then update cluster center//this loop is modifying k,dim in main for no reason
        for (int l = 0; l < k; ++l) {
            for (int i = 0; i < dim; ++i) {
                if(cluster_size[l] > 0){
                    cluster_center[(k*dim)+i] = array[(k*dim)+i]/cluster_size[l];
                }
                //reset array to 0 after updating cluster center
                array[(k*dim)+i] = 0;//this line modifes k after some iterations first to a random number then 0
            }
        }


        change /= N;
        loop_counter++;
    }while(loop_counter < 1500 && (change > threshold) );//do while there is change in clusters or reached a lot of iterations

    printf("loopcounter=%d\n",loop_counter);

    return 0;
}

int find_closest_cluster(int k, int dim, double *cluster_center,double *point){
    double distance, min_distance;
    int index, i;
    index=0;
    min_distance=euclidean_distance_8dims(point,cluster_center);
    for(i=dim;i<k*dim;i+=dim){
        distance=euclidean_distance_8dims(point,&cluster_center[i]);
        if(distance<min_distance){
            min_distance=distance;
            index=i;
        }
    }
    return index;
}

int choose_initial_centers(int dim, double *data, int k, double *cluster_center){
    //initialize cluster_center to -1
    for (int m = 0; m < k * dim; ++m) {
        cluster_center[m]=-1;
    }
    //get first cluster center randomly
    int x = (abs((rand() * dim)) % (N*dim));
    //loop to assign the center
    for (int i = 0; i < dim; ++i) {
        cluster_center[i] = data[i+x];
    }
    //to get the second center, find the farthest point from the intial center
    int p2 = find_farthest_point(dim,data,cluster_center);
    //loop to assign new center (2)
    for (int j = 8; j < dim*2; ++j) {
        cluster_center[j]=data[p2+j];
    }
    //to get centers from k=3 up to k centroids
    double max_mind,distance;
    int index,closest_cindex,closest_cluster;
    index=0;
    for (int l = 2; l < k ; ++l) {
        closest_cluster=find_closest_cluster(k,dim,cluster_center,data);
        max_mind=euclidean_distance_8dims(data,&cluster_center[closest_cluster]);
        for (int i = dim; i < N*dim; i+=dim) {
            closest_cindex=find_closest_cluster(k,dim,cluster_center,&data[i]);
            distance=euclidean_distance_8dims(&data[i],&cluster_center[closest_cindex]);
            if(distance>max_mind){
                max_mind=distance;
                index=i;
            }
        }
        //update new center
        for (int j = l*dim; j < (l*dim+dim); ++j) {
            cluster_center[j]=data[index];
            //increment index
            index++;
        }
        //at this point we have a new center, reset variables to get the next one, or put them inside outer for loop
        //reset index
        index=0;

    }

    return 0;
}

int find_farthest_point(int dim, double *data, double *point){
    double distance, max_distance;
    int index=0;
    max_distance=euclidean_distance_8dims(point,data);
    for (int i = dim; i < N * dim; i +=dim) {
        distance=euclidean_distance_8dims(point,&data[i]);
        if (distance > max_distance){
            max_distance=distance;
            index=i;
        }
    }
    return index;
}

int kmeans2(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size){
    int index;
    double change;
    int loop_counter=0;
    bool new_center;
    //double cluster_points[k][N*dim];
    double *test;
    test=calloc(k*N*dim, sizeof(double));//beta, will replace cluster_points in the future
    //p=&cluster_points[0][0];
//    for (int m = 0; m < k; ++m) {
//        for (int i = 0; i < N * dim; ++i) {
//            cluster_points[m][i] = 0;
//        }
//    }
    //trying book solution for initializing 2d array instead of for loop above
    //for (p; p < &cluster_points[k][N*dim]; ++p) {
    //    *p=0.0;
    //}

    double array[dim*k];
    //initialize array to 0
    memset(array, 0, dim*k*sizeof(double));
    //start with k random initial centers
//    for(int i=0;i<k*dim;i++){
//        cluster_center[i]=rand();

//    }
    //commented the above for loop to instead use the choose_initial_centers function to get cluster centers
    choose_initial_centers(dim,data,k,cluster_center);

    //initialize cluster_assign, using -1 to denote that the points have no cluster yet
    for(int i=0;i<N;i++){
        cluster_assign[i]=-1;
    }
    //initialize cluster size initially to 0
    for (int i = 0; i < k; i++) {
        cluster_size[i]=0;
    }
    //each data point calculates its distance to the k centers.
    //assign the data point to the cluster whose center is the closest
    int original_cluster;
    do{
        change=0;
        new_center=false;
        for (int g = 0; g < N * dim; g+=dim) {
            original_cluster=cluster_assign[g/dim];
            //find the array index of the closest cluster center
            index=find_closest_cluster(k,dim,cluster_center,&data[g]);
            //if index/dim != cluster_assign[g/dim] the point has a new cluster,
            //increment change variable & decrement size of old cluster, do nothing if it didnt have an old cluster
            if(index/dim != cluster_assign[g/dim]){
                change++;
                //assign membership of point, index/dim=k cluster
                cluster_assign[g/dim]=index/dim;
                //update size of the cluster
                cluster_size[index/dim]++;
                //update cluster_points
                for (int i = 0; i < dim; ++i) {
                    //cluster_points[index/dim][g+i] = data[g+i];
                    //beta test
                    test[(N*dim*index/dim)+i] = data[g+i];
                }
                if(original_cluster > -1 && original_cluster < k){ //if the point had a previous cluster, decrease its size
                    cluster_size[original_cluster]--;
                }

            }

        }
        //sum & avg to get new centers for each cluster
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < cluster_size[i] * dim; ++j) {
                //array[(i*dim) +(j%dim)]+=cluster_points[i][j];//sum the points
                //beta
                array[(i*dim) +(j%dim)]+=test[(i*dim*N)+j];
            }
            for (int l = 0; l < dim; ++l) {
                cluster_center[(i*dim)+l]=array[(i*dim)+l]/cluster_size[i];//get the avg & assign the new center
                //reset array for next iteration
                array[(i*dim)+l]=0;
            }
            //take care of empty clusters here, by using max_min procedure to get new center
            if(cluster_size[i]==0){
                max_min_newcenter(dim,data,k,cluster_center,i);
                new_center=true;
            }
        }

        change /= N;
        loop_counter++;
    }while( (change > threshold && loop_counter < 100) || (new_center && loop_counter < 200) );//do while there is change in clusters or reached a lot of iterations

    printf("loopcounter=%d\nThe quality of the clustering (SSE) is:= %lf\n",loop_counter,sum_of_squared_errors(dim,k,cluster_center,cluster_size,test));

    return 0;
}

double sum_of_squared_errors(int dim, int k, double *cluster_center, int *cluster_size, double *test){
    //function to measure  the quality of the clustering
    double sse, distance;
    sse=0;
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < cluster_size[j]; i+=dim) {
            //distance=euclidean_distance_8dims(&cluster_points[j][i],&cluster_center[j*dim]);
            //beta
            distance=euclidean_distance_8dims(&test[(j*dim*N)+i],&cluster_center[j*dim]);
            sse+=pow(distance,2);
        }
    }
    return sse;
}

int max_min_newcenter(int dim, double *data, int k, double *cluster_center,int in){
    double max_mind,distance;
    int index,closest_cindex,closest_cluster;
    index=0;
    closest_cluster=find_closest_cluster(k,dim,cluster_center,data);
    max_mind=euclidean_distance_8dims(data,&cluster_center[closest_cluster]);
    for (int i = dim; i < N*dim; i+=dim) {
        closest_cindex=find_closest_cluster(k,dim,cluster_center,&data[i]);
        distance=euclidean_distance_8dims(&data[i],&cluster_center[closest_cindex]);
        if(distance>max_mind){
            max_mind=distance;
            index=i;
        }
    }
    //update new center

    for (int x=in*dim; x < in*dim + dim; ++x) {
        cluster_center[x]=data[index];
        index++;
    }
    return 0;
}

int bisecting_kmeans(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size){
    double cluster_ssd[k];
    double *test;
    int max_clusterssd=-1;
    test=calloc(k*N*dim, sizeof(double));
    //initialize cluster_center to -1
    for (int m = 0; m < k * dim; ++m) {
        cluster_center[m]=-1;
    }
    //get first cluster center randomly
    int x = (abs((rand() * dim)) % (N*dim));
    //loop to assign the center
    for (int i = 0; i < dim; ++i) {
        cluster_center[i] = data[i+x];
    }
    //to get the second center, find the farthest point from the intial center
    int p2 = find_farthest_point(dim,data,cluster_center);
    //loop to assign new center (2)
    for (int j = 8; j < dim*2; ++j) {
        cluster_center[j]=data[p2+j];
    }
    //initialize cluster_assign, using -1 to denote that the points have no cluster yet
    for(int i=0;i<N;i++){
        cluster_assign[i]=-1;
    }
    //initialize cluster size initially to 0
    for (int i = 0; i < k; i++) {
        cluster_size[i]=0;
    }
    int count_k=2;
    //count_k is the number of clusters processed until the line of code
    //call 2-means to bisect
    //might need to enclose the lines below in a while loop to repeat until count_k == k
    //but be careful with the last parameter of the call to 2means as right now
    //this call to 2means will assign all points to their respective cluster and count_k should be two

    do{
        kmeans3(dim,data,2,cluster_assign,cluster_center,cluster_size,test,max_clusterssd,count_k);
        squared_standard_deviation(dim,k,cluster_center,cluster_size,test,cluster_ssd);
        max_clusterssd=max_index(cluster_ssd,k);
        count_k++;
    }while(count_k != k);

    //Now call kmeans2.2 with loop_counter limit = 5 and then we are done.
    kmeans2_2(dim,data,k,cluster_assign,cluster_center,cluster_size);

    return 0;
}

void squared_standard_deviation(int dim, int k, double *cluster_center, int *cluster_size, double *test,double *cluster_ssd){
    //function to get the squared standard deviation for each cluster
    double ssd, distance;
    ssd=0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < cluster_size[i]; j+=dim) {
            distance=euclidean_distance_8dims(&test[(i*dim*N)+j],&cluster_center[i*dim]);
            ssd+=pow(distance,2);
        }
        cluster_ssd[i]=ssd;
        ssd=0;
    }

    return;
}

int kmeans3(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size, double *test, int max_clusterssd, int count_k){
    int index,r,x,lower_bound,upper_bound, replace_point = -1;
    double change;
    int loop_counter=0;
    bool found_old_point;
    //bool new_center;
    double array[dim*count_k];
    found_old_point=false;
    //initialize array to 0
    memset(array, 0, dim*k*sizeof(double));
    //each data point calculates its distance to the k centers.
    //assign the data point to the cluster whose center is the closest
    int original_cluster;

    if(max_clusterssd == -1){
        // If this branch is taken it means we don't have any clusters assigned yet
        do{
            change=0;
//            new_center=false;
            for (int g = 0; g < N * dim; g+=dim) {
                original_cluster=cluster_assign[g/dim];
                //find the array index of the closest cluster center
                index=find_closest_cluster(k,dim,cluster_center,&data[g]);
                //if index/dim != cluster_assign[g/dim] the point has a new cluster,
                //increment change variable & decrement size of old cluster, do nothing if it didn't have an old cluster
                if(index/dim != cluster_assign[g/dim]){
                    change++;
                    //assign membership of point, index/dim=k cluster
                    cluster_assign[g/dim]=index/dim;
                    //update size of the cluster
                    cluster_size[index/dim]++;
                    //update cluster_points
                    //also need to reset values of a point that changed clusters (in test)
                    for (int i = 0; i < dim; ++i) {
                        test[(N*dim*index/dim)+(cluster_size[index/dim]*dim)+i] = data[g+i];
                    }
                    if(original_cluster > -1 && original_cluster < k){
                        //if the point had a previous cluster, decrease its size
                        cluster_size[original_cluster]--;
                    }
                }
            }
            //sum & avg to get new centers for each cluster
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < cluster_size[i] * dim; ++j) {
                    //array[(i*dim) +(j%dim)]+=cluster_points[i][j];//sum the points
                    //beta
                    array[(i*dim) +(j%dim)]+=test[(i*dim*N)+j];
                }
                for (int l = 0; l < dim; ++l) {
                    cluster_center[(i*dim)+l]=array[(i*dim)+l]/cluster_size[i];//get the avg & assign the new center
                    //reset array for next iteration
                    array[(i*dim)+l]=0;
                }
                //take care of empty clusters here, by using max_min procedure to get new center,
                //for this version of kmeans, I'll have to remove this part probably
//                if(cluster_size[i]==0){
//                    max_min_newcenter(dim,data,k,cluster_center,i);
//                    new_center=true;
//                }
            }

            change /= N;
            loop_counter++;
        }while( (change > threshold && loop_counter < 100) );//do while there is change in clusters or reached a lot of iterations

    }

    else{
        loop_counter=0;
        // If this branch is taken, it means we already have at least 2 clusters
        lower_bound=max_clusterssd*dim*N;
        upper_bound=(( (max_clusterssd*dim*N) + (cluster_size[max_clusterssd]*dim) ) -1);
        // Get a random point within the cluster about to be bisected.
        do{
            r = rand();
            // Keep getting random numbers until r is within bounds and the number is multiple of dim
        }while (r < lower_bound || r > upper_bound || (r%dim != 0));
        // Now find the furthest point from the random point (within the cluster)
        x = farthest_point_withinc(dim,test,&test[r],max_clusterssd,cluster_size);
        // Now update cluster centers
        for (int i = 0; i < dim; ++i) {
            cluster_center[max_clusterssd*dim + i] = test[r+i];
            cluster_center[(count_k-1)*dim + i] = test[x+i];
        }
        // Assign the points within the cluster to the newly assigned centers
        do{
            change=0;
//            new_center=false;
            for (int g = 0; g < N*dim; g+=dim){
                original_cluster=cluster_assign[g/dim];
                //find the array index of the closest cluster center
                index=find_closest_cluster(k,dim,cluster_center,&data[g]);
                //if index/dim != cluster_assign[g/dim] the point has a new cluster,
                //increment change variable & decrement size of old cluster, do nothing if it didn't have an old cluster
                if(index/dim != cluster_assign[g/dim] && (index/dim == max_clusterssd || index/dim == count_k-1)){
                    change++;
                    //assign membership of point, index/dim= k cluster
                    cluster_assign[g/dim]=index/dim;
                    //update size of the cluster
                    cluster_size[index/dim]++;
                    //update cluster_points (ie the test array)
                    //also need to reset values of a point that changed clusters (in test array too)
                    for (int i = 0; i < dim; ++i) {
                        test[(N * dim * index / dim) + (cluster_size[index / dim] * dim) + i] = data[g + i];
                        // Now need to remove the point I moved to a new place in test ie reset its old location
                        // Seems like it won't reset the newly added point because of the bounds in the for loop, but watch for this
                        // Another thing to watch is that the for loop going through test finds a duplicate point after already having a replace point,
                        //  could fix this with a break statement inside the if(memcmp), will apply this if it gives problems
                        if (!found_old_point) {
                            for (int j = N * dim * original_cluster; j < (dim * N * original_cluster) + (dim * cluster_size[original_cluster]); j += dim) {
                                if (memcmp(&data[g], &test[j], sizeof(double) * dim) == 0) {
                                    found_old_point = true;
                                    replace_point = j;
                                }
                            }
                        }
                        test[replace_point+i] = 0;
                    }
                    found_old_point=false;
                    if(original_cluster > -1 && original_cluster < k){
                        //if the point had a previous cluster, decrease its size
                        cluster_size[original_cluster]--;
                    }
                }
            }
            //sum & avg to get new centers for each cluster
            //make sure array is 0
            //reset array[20] for some reason at this point is not zero
            array[20] = 0;
            for (int j = 0; j < dim * cluster_size[max_clusterssd]; ++j) {
                   array[(max_clusterssd*dim) + (j%dim)]+=test[(max_clusterssd*dim*N)+j];
            }
            for (int l = 0; l < dim; ++l) {
                cluster_center[(max_clusterssd*dim)+l]=array[(max_clusterssd*dim)+l]/cluster_size[max_clusterssd];
                //get the avg & assign the new center
                //reset array for next iteration
                array[(max_clusterssd*dim)+l]=0;
            }
            for (int j = 0; j < dim * cluster_size[count_k-1]; ++j) {
                array[((count_k-1)*dim) + (j%dim)]+=test[((count_k-1)*dim*N)+j];
            }
            for (int l = 0; l < dim; ++l) {
                cluster_center[((count_k-1)*dim)+l]=array[((count_k-1)*dim)+l]/cluster_size[count_k-1];
                //get the avg & assign the new center
                //reset array for next iteration
                array[((count_k-1)*dim)+l]=0;
            }

            change /= N;
            loop_counter++;
        }while( (change > threshold && loop_counter < 100) );
    }
    //printf("loopcounter=%d\nThe quality of the clustering (SSE) is:= %lf\n",loop_counter,sum_of_squared_errors(dim,k,cluster_center,cluster_size,test));

    return 0;
}

int max_index(double *array,int array_size){
    // Function to get index of the biggest element in a double array.
    int max=0;
    for (int i = 1; i < array_size; ++i) {
        if(array[i] > array[max]){
            max=i;
        }
    }
    return max;
}

int farthest_point_withinc(int dim, double *test, double *point, int max_clusterssd, int *cluster_size){
    // Function to get farthest point within a cluster about to be bisected, note that the index
    // returned is relative to the test array not the data array
    double distance, max_distance;
    int index=max_clusterssd*dim*N;
    max_distance=euclidean_distance_8dims(&test[index], point);
    for (int i = index+dim; i < ( (index) + (cluster_size[max_clusterssd] * dim)); i+=dim) {
        distance=euclidean_distance_8dims(point,&test[i]);
        if (distance > max_distance){
            max_distance=distance;
            index=i;
        }
    }
    return index;
}

int kmeans2_2(int dim, double *data, int k, int *cluster_assign, double *cluster_center, int *cluster_size){
    //Will do 5 iterations to complete bisecting k-means.
    int index;
    double change;
    int loop_counter=0;
    //bool new_center;
    double *test;
    test=calloc(k*N*dim, sizeof(double));
    double array[dim*k];
    //initialize array to 0
    memset(array, 0, dim*k*sizeof(double));
    //each data point calculates its distance to the k centers.
    //assign the data point to the cluster whose center is the closest
    int original_cluster;
    do{
        change=0;
        //new_center=false;
        for (int g = 0; g < N * dim; g+=dim) {
            original_cluster=cluster_assign[g/dim];
            //find the array index of the closest cluster center
            index=find_closest_cluster(k,dim,cluster_center,&data[g]);
            //if index/dim != cluster_assign[g/dim] the point has a new cluster,
            //increment change variable & decrement size of old cluster, do nothing if it didnt have an old cluster
            if(index/dim != cluster_assign[g/dim]){
                change++;
                //assign membership of point, index/dim=k cluster
                cluster_assign[g/dim]=index/dim;
                //update size of the cluster
                cluster_size[index/dim]++;
                //update cluster_points
                for (int i = 0; i < dim; ++i) {
                    test[(N*dim*index/dim)+(cluster_size[index/dim]*dim)+i] = data[g+i];
                }
                if(original_cluster > -1 && original_cluster < k){
                    //if the point had a previous cluster, decrease its size
                    cluster_size[original_cluster]--;
                }
            }
        }
        //sum & avg to get new centers for each cluster
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < cluster_size[i] * dim; ++j) {
                array[(i*dim) +(j%dim)]+=test[(i*dim*N)+j];
            }
            for (int l = 0; l < dim; ++l) {
                cluster_center[(i*dim)+l]=array[(i*dim)+l]/cluster_size[i];//get the avg & assign the new center
                //reset array for next iteration
                array[(i*dim)+l]=0;
            }
            //take care of empty clusters here, by using max_min procedure to get new center
            if(cluster_size[i]==0){
                max_min_newcenter(dim,data,k,cluster_center,i);
                //new_center=true;
            }
        }

        change /= N;
        loop_counter++;
    }while( (change > threshold && loop_counter <= 5) );//do while there is change in clusters or reached 5 iterations

    //printf("loopcounter=%d\nThe quality of the clustering (SSE) is:= %lf\n",loop_counter,sum_of_squared_errors(dim,k,cluster_center,cluster_size,test));
    return 0;

}