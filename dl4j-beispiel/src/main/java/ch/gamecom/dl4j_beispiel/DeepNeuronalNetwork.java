package ch.gamecom.dl4j_beispiel;

import java.io.IOException;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DeepNeuronalNetwork extends LenetMnistExample {

    private static final Logger log = LoggerFactory.getLogger(LenetMnistExample.class);

    public static final int nChannels = 1; // Number of input channels
    public static final int outputNum = 10; // The number of possible outcomes
    public static final int batchSize = 64; // Test batch size
    public static final int nEpochs = 1; // Number of training epochs
    public static final int seed = 123; //

    public void init() throws IOException, InterruptedException {

        log.info("Load data....");
        DataSetIterator mnistTrain = loeadTrainingData();
        DataSetIterator mnistTest = loadTestData();

        log.info("Build model....");

        MultiLayerConfiguration conf = createNeuronalNetwork();
        MultiLayerNetwork model = createModel(conf);

        log.info("Train model....");

        trainModel(mnistTrain, mnistTest, model);
        log.info("****************Example finished********************");

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
