#include<cstdio>
#include<bitset>

#include <fstream>
#include <string>
// #define ONLINE_JUDGE
#define INPUT_DATA_TYPE int
#define OUTPUT_DATA_TYPE int
INPUT_DATA_TYPE read(){register INPUT_DATA_TYPE x=0;register char f=0,c=getchar();while(c<'0'||'9'<c)f=(c=='-'),c=getchar();while('0'<=c&&c<='9')x=(x<<3)+(x<<1)+(c&15),c=getchar();return f?-x:x;}void print(OUTPUT_DATA_TYPE x){register char s[20];register int i=0;if(x<0){x=-x;putchar('-');}if(x==0){putchar('0');return;}while(x){s[i++]=x%10;x/=10;}while(i){putchar(s[--i]+'0');}return;}

const int lsize=60000;
std::ifstream mnist_image("data/src/train-labels-idx1-ubyte", std::ios::in | std::ios::binary);

int gc(){
    int ans;
	mnist_image.read(reinterpret_cast<char*>(&ans), 1);
    return ans;
}

int nchar2num(int n){
    int i=0,ans=0;
    for(i=0;i<n;++i){
        ans=(ans*256)+((unsigned char)gc());
    }
    return ans;
}

void getLables(){
    register int i=0,j=0;
    printf("%d",nchar2num(1));
}

int main(){
	#ifndef ONLINE_JUDGE
	freopen("name.in", "r", stdin);
	freopen("name.out", "w", stdout);
	#endif

    freopen("train_lables", "w", stdout);

    register int i;

    nchar2num(4);
    nchar2num(4);
    
    for(i=0;i<lsize;++i){
        getLables();putchar('\n');
    }
    print(lsize);putchar('\n');

	#ifndef ONLINE_JUDGE
	fclose(stdin);
	fclose(stdout);
	#endif
    return 0;
}