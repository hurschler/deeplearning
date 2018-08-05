package ch.hslu.prg.array;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ArraysZusammenZaehlen {

	public static void main(String[] args) {
		ArraysZusammenZaehlen azz = new ArraysZusammenZaehlen();
		int[] a = {1,2,3};
		int[] b = {5,6};		
		int[] ausgabe = azz.add(a, b);
		System.out.println(Arrays.toString(ausgabe));
	}

	public int[] add(int[] a, int[] b) {
		int xa = a.length;
		int xb = b.length;
		int n;
		if (xa < xb) {
			n = xa;
		} else {
			n = xb;
		}
		
		int[] r = new int[n];
		
		for (int i =0; i < n; i++) {
			r[i] = a[i] + b[i];
		}
		return r;
		
	}
	
	
}
