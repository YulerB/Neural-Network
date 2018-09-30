using System.Collections.Generic;

namespace NeuralNetwork.Helpers
{
    public class HelperNetwork
	{
        public double LearnRate { get; private set; }
		public double Momentum { get; private set; }
		public List<HelperNeuron> InputLayer { get; private set; }
		public List<List<HelperNeuron>> HiddenLayers { get; private set; }
		public List<HelperNeuron> OutputLayer { get; private set; }
		public List<HelperSynapse> Synapses { get; private set; }

        public HelperNetwork(double learnRate, double momentum, List<HelperNeuron> hnInputLayer, List<List<HelperNeuron>> hnHiddenLayers, List<HelperNeuron> hnOutputLayer, List<HelperSynapse> hnSynapses)
        {
            this.LearnRate = learnRate;
            this.Momentum = momentum;
            this.InputLayer = hnInputLayer;
            this.HiddenLayers = hnHiddenLayers;
            this.OutputLayer = hnOutputLayer;
            this.Synapses = hnSynapses;
        }
    }
}
