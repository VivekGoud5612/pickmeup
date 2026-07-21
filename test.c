#include<stdio.h>

void main(){
    int a = 2, b=27;
    int res=0;
    int rem = b;
    while(rem - a>= 0){
        rem = rem - a;
        res++;
    }
    printf("%d", res);
    printf(".");
    for(int i = 0; i<4; i++){
        rem = rem * 10;
        int digit = 0;
        while(rem - a>= 0){
            rem = rem - a;
            digit++;
        }
        printf("%d",digit);
    }
}




recruit@gridbots.com