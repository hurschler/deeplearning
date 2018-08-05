package ch.hslu.prg.simon;

public class MainKlasse {

	public static void main(String[] args) {
		
		Mitarbeiter m = new Dozent();
		var p1 = (Person) m;
		
		Dozent d2 = (Dozent) new Person();
		Student etStudent = new EtStudent();
	}

}
