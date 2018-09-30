using System.Collections.Generic;

namespace NeuralNetwork.Helpers
{
    public class HelperNetwork
	{
        public double LearnRate { get;  set; }
		public double Momentum { get;  set; }
		public List<HelperNeuron> InputLayer { get;  set; }
		public List<List<HelperNeuron>> HiddenLayers { get;  set; }
		public List<HelperNeuron> OutputLayer { get;  set; }
		public List<HelperSynapse> Synapses { get;  set; }

        public HelperNetwork() { }
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
