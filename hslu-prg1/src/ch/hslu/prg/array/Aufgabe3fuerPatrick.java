package ch.hslu.prg.array;

import java.awt.Color;
import java.util.HashMap;
import java.util.Iterator;

public class Aufgabe3fuerPatrick {

	public static void main(String[] args) {
		
		HashMap<String,Color> m;
		m = new HashMap<>();
		m.put("green", Color.green);
		
		Iterator<String> itr = m.keySet().iterator();
		while (itr.hasNext()) {
			String key = itr.next();
			Color c = m.get(key);
			System.out.println("Schluessel:" + key + " Farbe:" + c.toString());
		}

	}

}
