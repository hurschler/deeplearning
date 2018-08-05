package ch.hslu.prg.array;

import static org.junit.Assert.*;

import org.junit.Test;

public class Vergleich {

	@Test
	public void test() {
		Integer a = new Integer(999);
		Integer b = new Integer(999);
		
		System.out.println(a == b);
		System.out.println(a.equals(b));
		
		
	}

}
