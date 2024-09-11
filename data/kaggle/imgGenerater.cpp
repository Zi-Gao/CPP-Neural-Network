#include<iostream>
#include<cstdio>
#include<vector>
#include<random>
#include<algorithm>

struct PIC{
    int pic[28][28],res;
};

std::vector<PIC> pics;

int main(){
    std::mt19937 mt(time(0));

    int i,j,k,dx,dy;
    
    int n=60000;
    int more=20000;
    std::uniform_int_distribution<int> rdx(-2,2);
    std::uniform_int_distribution<int> rdy(-2,2);
    
    pics.resize(n);

    freopen("data/src/train_images","r",stdin);
    for(i=0;i<n;++i){
        for(j=0;j<28;++j)
            for(k=0;k<28;++k)
                scanf("%d",&pics[i].pic[j][k]);
    }
    freopen("data/src/train_lables","r",stdin);
    for(i=0;i<n;++i) scanf("%d",&pics[i].res);

    // std::shuffle(pics.begin(),pics.end(),mt);

    for(i=0;i<more;++i){
        dx=rdx(mt);
        dy=rdy(mt);
        PIC now=pics[i];
        for(j=0;j<28;++j)
            for(k=0;k<28;++k){
                now.pic[j][k]=0;
                if(j+dx<0||j+dx>28) continue;
                if(k+dy<0||k+dy>28) continue;
                now.pic[j][k]=pics[i].pic[j+dx][k+dy];
            }
        pics.push_back(now);
    }

    // pics.erase(pics.begin(),pics.begin()+n);

    std::shuffle(pics.begin(),pics.end(),mt);

    freopen("train_images","w",stdout);

    for(auto pic:pics){
        for(j=0;j<28;++j,putchar('\n'))
            for(k=0;k<28;++k){
                printf("%4d",pic.pic[j][k]);
            }
        putchar('\n');
    }

    printf("%d",pics.size());

    freopen("train_lables","w",stdout);

    for(auto pic:pics) printf("%d\n",pic.res);

    printf("%d",pics.size());

    return 0;
}