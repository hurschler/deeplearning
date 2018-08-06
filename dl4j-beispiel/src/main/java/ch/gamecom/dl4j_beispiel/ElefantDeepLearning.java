package ch.gamecom.dl4j_beispiel;

import java.io.IOException;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.io.ClassPathResource;

public class ElefantDeepLearning extends Iris2NeuralNetwork {

    public static void main(String[] args) throws IOException {

        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(new ClassPathResource("20171220_194540.jpg").getFile());

        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);

        // Inference returns array of INDArray, index[0] has the predictions
        INDArray[] output = vgg16.output(false, image);

        // convert 1000 length numeric index of probabilities per label
        // to sorted return top 5 convert to string using helper function
        // VGG16.decodePredictions
        // "predictions" is string of our results
        String predictions = TrainedModels.VGG16.decodePredictions(output[0]);
        System.out.println(predictions);
    }

}
