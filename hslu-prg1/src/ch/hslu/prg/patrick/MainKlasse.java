package ch.hslu.prg.patrick;

import java.util.Arrays;
import java.util.List;

import ch.hslu.prg.array.Simon;

public class MainKlasse {

	enum SimonsEnum {
		TA(1), MUSIK(2), INFORMATIK(3);
		int number;	
		
		SimonsEnum(int i) { 
			number = i;
		}
		
		@Override
		public String toString() {
			return  this.name() + " Number: " +number;
		}
			
		void ausgabe() {
			SimonsEnum.values();
		}
	}
	
	public static void main(String[] args) throws Exception {
		Tier tier = new Papagei();
		tier.essen();
		float f = 1.0f;
		f++;
		System.out.println(f);
		double d = 3.0;
		double r = 10.0d / d;
		
		float af;
		af = 3.01f;
		
		int i = 1;
		float bf = (float) ((float)i + d);
		
		long l = 222222222222222222l;
		double dd = l;
		
		System.out.println(dd);
		
		Math.random() ;
				
		System.out.println(r);
		
		MainKlasse.SimonsEnum simonsEnum = MainKlasse.SimonsEnum.INFORMATIK;
		
		
		switch (simonsEnum) {
		case TA:	
			break;

		default:
			break;
		}
		
		System.out.println();
		
		int[] q = new int[10]; 
		int b[] = new int[10]; 
		
		short s = 127;
		byte bl = 127;
		System.out.println("short:" + s);
		System.out.println("byte:" + bl);
		
		System.out.println("Int Array q " + q);
		System.out.println("Int Array b " + b);
		
		for (int qInt  : b) {
			System.out.println("List of q's: " + qInt);
		}
		
		for (SimonsEnum js : SimonsEnum.values()) {
			System.out.println("SimonsEnum Value: " + js);
		}
		
		
	}

}
