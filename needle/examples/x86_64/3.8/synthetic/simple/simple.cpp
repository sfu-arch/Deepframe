#include <stdio.h>
#include <stdlib.h>

int size = 5000;
int* a = NULL;

void func () {


	for (int i = 0; i < 100000; i ++) {

		int now = i % (size - 1) + 1 ;
		int prev = now - 1;

		//a[now] = 5 + a[prev] + a[0];

		if (i % 2 == 0) {
			a[now] = 5 + a[prev];

		} else {

			a[now] = 10 + a[prev];
		}

	}
}


int main() {

	a = new int [size];

	func();

	printf("midpoint = %d\n", a[size/2]);	

	return 0;

}
