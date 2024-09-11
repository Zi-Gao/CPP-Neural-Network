#include<bits/stdc++.h>
// #define ONLINE_JUDGE
#define INPUT_DATA_TYPE int
#define OUTPUT_DATA_TYPE int
INPUT_DATA_TYPE read(){INPUT_DATA_TYPE x=0;char f=0,c=getchar();while(c<'0'||'9'<c)f=(c=='-'),c=getchar();while('0'<=c&&c<='9')x=(x<<3)+(x<<1)+(c&15),c=getchar();return f?-x:x;}void print(OUTPUT_DATA_TYPE x){char s[20];int i=0;if(x<0){x=-x;putchar('-');}if(x==0){putchar('0');return;}while(x){s[i++]=x%10;x/=10;}while(i){putchar(s[--i]+'0');}return;}


const int psize=42000;

int main(){
    freopen("train.csv", "r", stdin);
    freopen("train_images", "w", stdout);
    

    int i,x,y;
    char c=getchar();
    while(c!='\n') c=getchar();
    for(i=0;i<psize;++i,putchar('\n')) for(read(),x=0;x<28;++x,putchar('\n')) for(y=0;y<28;++y) printf("%3d ",read());
    print(psize);

	#ifndef ONLINE_JUDGE
	fclose(stdin);
	fclose(stdout);
	#endif
    return 0;
}