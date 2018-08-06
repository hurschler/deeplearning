package ch.gamecom.dl4j_beispiel;

public class StartDeepLearningNetwork extends DeepNeuronalNetwork {
    public static void main(String[] args) throws Exception {
        // DeepNeuronalNetwork dnn = new DeepNeuronalNetwork();
        IrisDeepNeuronalNetwork dnn = new IrisDeepNeuronalNetwork();
        dnn.init();
    }
}
