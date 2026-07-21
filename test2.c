#include<stdio.h>
int seed;
unsigned int generate_seed(unsigned int seed){
    seed = seed * 1106715286 + 12345;
    return (seed)%2147483647;
}
void main(){
    int arr[100];
    int dones[100];
    for(int i = 0; i < 100; i++){
        arr[i] = i+1;
        dones[i] = 0;
    }
    int index;
    int rand_factor = 3343;
    for(int i = 0; i < 100; i++){
        rand_factor = generate_seed(rand_factor);
        index = rand_factor % 100;
        while(dones[index] == 1){
            rand_factor = generate_seed(rand_factor);
            index = rand_factor % 100;
        }
        dones[index] = 1;
        printf("\n");
        if(dones[index] == 0) printf("hi");
        else printf("%d", arr[index]);
        
    }
}