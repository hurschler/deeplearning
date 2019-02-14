package com.fenaco.ua.ki;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.gamecom.dl4j_beispiel.IrisDeepNeuronalNetwork;

public class KuhKITrainer extends IrisDeepNeuronalNetwork {
    private static Logger log = LoggerFactory.getLogger(KuhKITrainer.class);

    public KuhKITrainer() {
    	
    };
    
    public static void main(String[] args) throws Exception {

    	// Konfiguration
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        
        KuhKITrainer kuhKiTrainer = new KuhKITrainer();
        // Daten der Kühe vorbereiten
        DataSet trainingData = kuhKiTrainer.datenVorbereiten();
        
        // Neuronales Netz definieren
        MultiLayerNetwork model = kuhKiTrainer.modelDefinieren();
        
        // Jetzt das Netz mit den Kühen trainieren ..... Muuuhhhh
        kuhKiTrainer.modelTrainieren(trainingData, model);
        
        // das fertige Model ins Dateisystem speichern
        kuhKiTrainer.modelSpeichern(model);
        
        
        log.info("bin fertig mit trainieren");
        
    }

	private void modelSpeichern(MultiLayerNetwork model) throws IOException {
		File file = new File("kuh-model.zip");
        ModelSerializer.writeModel(model, file, true);
	}

	private void modelTrainieren(DataSet trainingData, MultiLayerNetwork model) {
		for (int i = 0; i < 1000; i++) {
            model.fit(trainingData);
        }
	}

	private MultiLayerNetwork modelDefinieren() {
		final int numInputs = 4;
        int outputNum = 3;
        long seed = 6;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
        		.activation(Activation.TANH)
        		.weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1)).l2(1e-4).list().layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
                .backprop(true).pretrain(false).build();
       
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
		return model;
	}

	private DataSet datenVorbereiten() throws IOException, InterruptedException {
		int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("kuhdatenbank.txt").getFile()));

        int labelIndex = 4; 
        int numClasses = 3; 
        int batchSize = 150; 

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8); 

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();


        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData); 
        normalizer.transform(trainingData); 
        normalizer.transform(testData);
		return trainingData;
	}
}
