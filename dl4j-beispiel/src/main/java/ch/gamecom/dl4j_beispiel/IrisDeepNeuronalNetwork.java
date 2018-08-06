package ch.gamecom.dl4j_beispiel;

import java.io.IOException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IrisDeepNeuronalNetwork extends DeepNeuronalNetwork {
    private static final Logger log = LoggerFactory.getLogger(LenetMnistExample.class);

    public static final int nChannels = 1; // Number of input channels
    public static final int outputNum = 10; // The number of possible outcomes
    public static final int batchSize = 64; // Test batch size
    public static final int nEpochs = 1; // Number of training epochs
    public static final int seed = 123; //

    public static final int CLASSES_COUNT = 3;
    public static final int FEATURES_COUNT = 4;

    @Override
    public void init() throws IOException, InterruptedException {

        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
            DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            DataSet allData = iter.next();
            allData.shuffle();

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER).list()
                    .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build()).layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3)
                            .nOut(CLASSES_COUNT).build())
                    .backprop(true).pretrain(false).build();

            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            model.init();
            model.fit(trainingData);

            INDArray output = model.output(testData.getFeatureMatrix());
            Evaluation eval = new Evaluation(3);
            eval.eval(testData.getLabels(), output);

        }

    }

    private void trainModel(DataSetIterator mnistTrain, DataSetIterator mnistTest, MultiLayerNetwork model) {
        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = model.evaluate(mnistTest);
            log.info(eval.stats());
            mnistTest.reset();
        }
    }

    private MultiLayerNetwork createModel(MultiLayerConfiguration conf) {
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // Print score every
        // 10 iterations
        return model;
    }

    private MultiLayerConfiguration createNeuronalNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).l2(0.0005).weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9)).list().layer(0, new ConvolutionLayer.Builder(5, 5)
                        // nIn and nOut specify depth. nIn here is the nChannels
                        // and nOut is the number of filters to be applied
                        .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build()).layer(2, new ConvolutionLayer.Builder(5, 5)
                        // Note that nIn need not be specified in later layers
                        .stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1)) // See
                                                                      // note
                                                                      // below
                .backprop(true).pretrain(false).build();
        return conf;
    }

    private DataSetIterator loadTestData() throws IOException {
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);
        return mnistTest;
    }

    private DataSetIterator loeadTrainingData() throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        return mnistTrain;
    }

}
