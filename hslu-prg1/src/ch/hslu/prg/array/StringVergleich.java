package ch.hslu.prg.array;

import static org.junit.Assert.*;

import org.junit.Test;

public class StringVergleich {

	String s = "";
	int i = 0;
	
	public String stringAnhaengen(String aneinanderReihen) {		
		if (i < 3) {
			s = s + aneinanderReihen;
			i++;
			stringAnhaengen(s);
		} else {
		  return s;
		}
		return s;
	}
	
	@Test
	public void test() {
		System.out.println(stringAnhaengen(" gugus"));
	}

}
