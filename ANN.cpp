#include<cstdio>
#include<cmath>
#include<vector>
#include<cstdlib>
#include<random>
#include<assert.h>
#include<thread>
#include<iostream>
#include<cstring>
// #include<Eigen/Dense>

typedef float fp;

#define inl inline __attribute((always_inline))
// #define inl

/*
INPUT_SIZE:  输入向量长度
OUTPUT_SIZE: 输出向量长度
NET_DEEP:    中间层深度
NET_SIZE:    中间层向量长度
*/
const int PIC_SIZE=28,INPUT_SIZE=784,OUTPUT_SIZE=10,NET_DEEP=16,NET_SIZE=1000,MAX_PIC=1000010;
const int MAX_BATCH=64;
std::mt19937 mt(time(0));

int net_sizes[100]={0,
    784,600,450,200,
    128,128,128,128,
    128,128,128,128,
    128,128,128,128,
};

inl float sigmoid(float x,char flg) {
    if(flg) return sigmoid(x,0)*(1-sigmoid(x,0));
	return (float)(1.0)/(1+std::exp(-x));
}
inl float swish(float x,char flg) {
    if(flg) return sigmoid(x,0)*(1-swish(x,0))+swish(x,0);
	return x*sigmoid(x,0);
}
inl float ReLU(float x,char flg){
    if(flg) return x<0?0:1;
    return x<0?0:x;
}
const float B_ELU=1;
inl float ELU(float x,char flg){
    if(flg) return x<0?(std::exp(x)*B_ELU):1;
    return x<0?((std::exp(x)-1)*B_ELU):x;
}
const float B_LeakyReLU=0.01;
inl float LeakyReLU(float x,char flg){
    if(flg) return x<0?B_LeakyReLU:1;
    return x<0?x*B_LeakyReLU:x;
}

inl float square(float x){return x*x;}

inl float logistic(float x){return ReLU(x,0);}
inl float dlogistic(float x){return ReLU(x,1);}

struct MATRIX{
    int n,m;
    fp **mat;

    MATRIX(){
        mat=NULL;
        n=m=-1;
    }

    ~MATRIX(){
        del();
    }

    MATRIX(const MATRIX &o){
        n=m=-1;
        int i,j;
        nw(o.n,o.m);
        memcpy(mat[0],o.mat[0],sizeof(fp)*n*m);
    }

    inl void del(){
        if(n!=-1){
            int i;
            delete[] mat[0];
            delete[] mat;
            mat=NULL;
            n=m=-1;
        }
    }

    inl void nw(int _n,int _m){
        del();
        int i;
        n=_n,m=_m;
        fp *p=new fp[n*m];
        mat=new fp*[n];
        mat[0]=p;
        for(i=1;i<n;++i) mat[i]=p+i*m;
    }

    inl void operator = (const MATRIX &o){
        int i,j;
        nw(o.n,o.m);
        memcpy(mat[0],o.mat[0],sizeof(fp)*n*m);
    }

    /*
    flg=0  set0
    flg=1 set random [a,b]
    flg=2 norm 
    */
    inl void init(int _n,int _m,char flg,float a=0,float b=0){
        int i,j;
        nw(_n,_m);
        if(flg==0){
            memset(mat[0],0,sizeof(fp)*n*m);
        }else if(flg==1){
            std::uniform_real_distribution<float> rd1(a,b);
            for(i=0;i<n*m;++i) mat[0][i]=rd1(mt);
        }else{
            std::normal_distribution<float> rd2(a,b);
            for(i=0;i<n*m;++i) mat[0][i]=rd2(mt);
        }

    }

    inl MATRIX operator + (const MATRIX &a) const {
        int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        assert(n==a.n&&m==a.m);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=mat[i][j]+a.mat[i][j];

        return ans;
    }

    inl MATRIX operator - (const MATRIX &a) const {
        int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        assert(n==a.n&&m==a.m);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=mat[i][j]-a.mat[i][j];

        return ans;
    }

    inl MATRIX operator * (const MATRIX &a) const {
        int i,j,k;
        MATRIX ans;
        ans.init(n,a.m,0);
        assert(m==a.n);
        for(i=0;i<n;++i)
            for(k=0;k<m;++k)
                for(j=0;j<a.m;++j)
                    ans.mat[i][j]+=mat[i][k]*a.mat[k][j];
        return ans;
    }

    inl MATRIX operator * (const fp &a) const {
        int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=mat[i][j]*a;
        return ans;
    }

    inl MATRIX operator ^ (const MATRIX &a) const {
        int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        assert(n==a.n&&m==a.m);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=mat[i][j]*a.mat[i][j];
        return ans;
    }

    inl MATRIX T() const{
        int i,j;
        MATRIX ans;
        ans.init(m,n,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[j][i]=mat[i][j];
        return ans;
    }

    inl MATRIX logis() const{
        int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=logistic(mat[i][j]);
        return ans;
    }

    inl MATRIX dlogis() const{
        int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=dlogistic(mat[i][j]);
        return ans;
    }

    inl void print() const{
        int i,j;
        for(i=0;i<n;++i,putchar('\n'))
            for(j=0;j<m;++j,putchar(' ')){
                printf("%.10f",(float)mat[i][j]);
            }
    }

    inl void prtMat() const{
        int i,j;
        float res;
        for(i=0;i<n;++i,putchar('\n'))
            for(j=0;j<m;++j,putchar(' '))
                res=mat[i][j],printf("%u",*(unsigned int*)&res);
    }

    inl void getMat(){
        int i,j;
        unsigned int x;
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                scanf("%u",&x),mat[i][j]=*(float*)&x;
    }
};

inl float getCost();
int getOutputAns(MATRIX output);
inl void work(int id,char flg);
inl void addDelta(int id,float w);
void getPic(int n,std::string lables,std::string images);
inl std::pair<MATRIX,int> getData();
void initBatch(int id);
void trainBatch(int lid,int rid);
void train(int data_size,int iters,int batch_size,float learn,int thd,int steps_per_log,int steps_per_clear,int steps_per_save);
void test(int data_size,int iters,int steps_per_log);
inl void init();
inl void save();
inl void getNet();

MATRIX B[NET_DEEP+10],W[NET_DEEP+10];
MATRIX Z[MAX_BATCH][NET_DEEP+10],DW[MAX_BATCH][NET_DEEP+10],DZ[MAX_BATCH][NET_DEEP+10];
MATRIX batch_input[MAX_BATCH],batch_output[MAX_BATCH];
int batch_ans[MAX_BATCH],batch_res[MAX_BATCH];
MATRIX inputPic[MAX_PIC];
int res[MAX_PIC],now=0,cntPic;

inl void getDelta(int id){
    int l,i,j;
    DZ[id][NET_DEEP+1]=(batch_output[id]-Z[id][NET_DEEP+1])*2;
    for(l=NET_DEEP;l;--l){
        for(i=0;i<W[l].n;++i)
            for(j=0;j<W[l].m;++j)
                DW[id][l].mat[i][j]=W[l].mat[i][j]*DZ[id][l+1].mat[0][j];
        DZ[id][l]=Z[id][l].dlogis()^(DZ[id][l+1]*(W[l].T()));
    }
    for(i=0;i<W[0].n;++i)
        for(j=0;j<W[0].m;++j)
            DW[id][0].mat[i][j]=W[0].mat[i][j]*DZ[id][1].mat[0][j];
    return;
}
inl void addDelta(int id,float w){
    int i;
    W[0]=W[0]+(DW[id][0]*w);
    for(i=1;i<=NET_DEEP;++i){
        B[i]=B[i]+(DZ[id][i]*w);
        W[i]=W[i]+(DW[id][i]*w);
    }

    return;
}

void initBatch(int id){
    ++now;
    now%=cntPic;
    batch_input[id]=inputPic[now];
    batch_output[id].init(1,10,0);
    batch_output[id].mat[0][batch_ans[id]=res[now]]=1;
    return;
}

inl float getCost(MATRIX output,MATRIX ans){
    int i;
    float res=0;
    for(i=0;i<OUTPUT_SIZE;++i) res+=square(ans.mat[0][i]-output.mat[0][i]);
    return res;
}

int getOutputAns(MATRIX output){
    int i,res=0;
    for(i=0;i<10;++i)
        if(output.mat[0][i]>output.mat[0][res]) res=i;
    return res;
}

inl void init(){
    printf("init net\n");

    int i,id;

    net_sizes[0]=INPUT_SIZE;
    net_sizes[NET_DEEP+1]=OUTPUT_SIZE;

    for(i=0;i<=NET_DEEP;++i){
        W[i].init(net_sizes[i],net_sizes[i+1],2,0,std::sqrt(2.0/net_sizes[i]));
        for(id=0;id<MAX_BATCH;++id) DW[id][i]=W[i];
    }
    for(i=1;i<=NET_DEEP;++i){
        B[i].init(1,net_sizes[i],0,0.001,0.001);
        for(id=0;id<MAX_BATCH;++id) Z[id][i]=B[i];
    }
    return;
}

void getPic(int n,std::string lables,std::string images){
    float x;

    printf("get pic\n");
    std::cout<<lables<<' '<<images<<'\n';

    cntPic=n,now=0;
    int i=0,j;
    freopen(lables.c_str(), "r", stdin);
    for(i=0;i<cntPic;++i) scanf("%d",res+i);
    freopen(images.c_str(), "r", stdin);
    for(i=0;i<cntPic;++i){
        inputPic[i].init(1,INPUT_SIZE,0);
        for(j=0;j<INPUT_SIZE;++j) scanf("%f",&x),inputPic[i].mat[0][j]=x/255;
    }
}
inl void save(){
    printf("save model\n");

    freopen("result", "w", stdout);
    int i;
    W[0].prtMat();
    for(i=1;i<=NET_DEEP;++i)
        W[i].prtMat();
    for(i=1;i<=NET_DEEP;++i)
        B[i].prtMat();
    freopen("/dev/tty", "w", stdout);
    return;
}
inl void getNet(){
    printf("get model from file\n");

    freopen("result", "r", stdin);
    int i;
    W[0].getMat();
    for(i=1;i<=NET_DEEP;++i) W[i].getMat();
    for(i=1;i<=NET_DEEP;++i) B[i].getMat();
    return;
}

inl void work(int id,char flg){
    int i;
    Z[id][1]=batch_input[id]*W[0];
    for(i=1;i<=NET_DEEP;++i){
        Z[id][i]=Z[id][i]+B[i];
        Z[id][i+1]=(Z[id][i].logis())*W[i];
    }
    batch_res[id]=getOutputAns(Z[id][NET_DEEP+1]);
}

void trainBatch(int lid,int rid){
    int i;
    for(i=lid;i<rid;++i){
        work(i,0);
        getDelta(i);
    }
    return;
}
const double PI=acos(-1);
void train(int data_size,int iters,int batch_size,float learn,int thd,int steps_per_log,int steps_per_clear,int steps_per_save){
    printf("start training\n");

    getPic(data_size,"train_lables","train_images");

    thd=std::min(thd,batch_size);

    int i,j,cntRight=0,cntAll=0,thd_s=batch_size/thd,now;
    double t;
    float learnNow=learn;
    time_t start=time(0),nowtime;

    std::vector<int> thd_size;
    std::vector<std::thread> thd_pool(thd);
    for(i=0;i<thd;++i) thd_size.push_back(thd_s);
    for(i=0;i<batch_size%thd;++i) thd_size[i]+=1;

    for(i=1;i<=iters;++i){
        if(i%steps_per_clear==0) cntRight=cntAll=0;

        for(j=0;j<batch_size;++j) initBatch(j);
        for(j=now=0;j<thd;++j){
            thd_pool[j]=std::thread(trainBatch,now,now+thd_size[j]);
            now+=thd_size[j];
        }
        for(j=0;j<thd;++j) thd_pool[j].join();

        // learnNow=(std::sin(i*1.0/10)+1.0)/2*learn;
        // learnNow=(0.01-0.05)/iters*i+0.05;
        // learnNow=(0.05-0.01)/iters*i+0.01;
        
        // learnNow=(0.005-0.05)/iters*i+0.05;
        // learnNow=(0.05-0.005)/iters*i+0.005;

        learnNow=(std::cos((i+iters-1)*PI/2/iters)+1)*learn+0.005;

        for(j=0;j<batch_size;++j) addDelta(j,learnNow/batch_size),cntRight+=(batch_ans[j]==batch_res[j]);
        cntAll+=batch_size;
        
        if(i%steps_per_log==0){
            nowtime=time(0)-start;
            t=nowtime*1.0/60;

            batch_output[0].print();
            Z[0][NET_DEEP+1].print();

            printf("cost:%f\n",getCost(batch_output[0],Z[0][NET_DEEP+1]));

            printf("train cases:%d train iters:%d train epoches:%lf\nlearn:%f\naccuracy rate:%lf\nuse time:%.3lfmins\nall time:%.3lfmins\nleft time:%.3lfmins\n",
                i*batch_size,i,i*batch_size*1.0/data_size,
                learnNow,
                cntRight*1.0/cntAll,
                t,t/i*iters,t/i*iters-t
            );
        }
        if(i%steps_per_save==0) save();
    }
}

void test(int data_size,int iters,int steps_per_log){
    printf("start testing\n");

    getPic(data_size,"train_lables","train_images");

    int i=0,cntRight=0,cntAll=0;
    time_t start=time(0),nowtime;
    double t;
    MATRIX input,ans;
    
    for(i=1;i<=iters;++i){
        initBatch(0);
        work(0,0);

        cntRight+=(batch_ans[0]==batch_res[0]);
        ++cntAll;
        if(i%steps_per_log==0){
            nowtime=time(0)-start;
            t=nowtime*1.0/60;
            printf("test case%d\naccuracy rate:%lf\nuse time:%.3lfmins all time:%.3lfmins left time:%.3lfmins\n",
                i,cntRight*1.0/cntAll,
                t,t/i*iters,t/i*iters-t
            );
        }
    }
    return;
}

int main(){
    int i;
    MATRIX input,output,ans;
    
    init();
    
    getNet();

    train(
        /*data_size*/       42000,
        /*iters*/           42000*4,
        /*batch_size*/      16,
        /*learn*/           0.01,
        /*thread*/          4,
        /*steps pre log*/   50,
        /*steps pre clear*/ 1000,
        /*steps pre save*/  1000
    );

    test(42000,42000,10);

    save();

    return 0;
}