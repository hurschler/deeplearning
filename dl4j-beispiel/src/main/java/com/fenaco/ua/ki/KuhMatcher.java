package com.fenaco.ua.ki;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KuhMatcher {

	private static Logger log = LoggerFactory.getLogger(KuhMatcher.class);
	
	public KuhMatcher() {
		
	}
	
    public static void main(String[] args) throws IOException, InterruptedException {

    	KuhMatcher kuhMatcher = new KuhMatcher();
    	
    	// Wir laden das gespeicherte Model
    	MultiLayerNetwork model = kuhMatcher.modelLaden();    	    	
    	     
    	// Wir lesen die Daten unserer noch nicht zugeordneten Kühe ein
        DataSet allData = kuhMatcher.datenDieWirPruefenWollenLaden();     

        // Und jetzt mit der KI die Kühe zuordnen
        kuhMatcher.welcheKuhgehoertWelchemBauer(model, allData);
    	
    }

	private void welcheKuhgehoertWelchemBauer(MultiLayerNetwork model, DataSet allData) {
		// evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(allData.getFeatures());        
        eval.eval(allData.getLabels(), output);
        System.out.println("Labels: \n " + allData.getLabels().toString()); 
        log.info(eval.stats());
	}

	private DataSet datenDieWirPruefenWollenLaden() throws IOException, InterruptedException {
		RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(new ClassPathResource("kuhdatenbank-small.txt").getFile()));      
        int labelIndex = 4; 
        int numClasses = 3; 
        int batchSize = 10; 

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData); 
        normalizer.transform(allData);
		return allData;
	}

	private MultiLayerNetwork modelLaden() throws IOException {
		File file = new File("kuh-model.zip");
    	MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(file);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

		return model;
	}

}
