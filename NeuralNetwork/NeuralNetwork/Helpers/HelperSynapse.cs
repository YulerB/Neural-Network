using System;

namespace NeuralNetwork.Helpers
{
    public class HelperSynapse
	{
        public HelperSynapse() { }
        public HelperSynapse(Guid id, Guid inputNeuronId, Guid outputNeuronId, double weight, double weightDelta)
        {
            this.Id = id;
            this.InputNeuronId = inputNeuronId;
            this.OutputNeuronId = outputNeuronId;
            this.Weight = weight;
            this.WeightDelta = weightDelta;
        }

        public Guid Id { get;  set; }
		public Guid OutputNeuronId { get;  set; }
		public Guid InputNeuronId { get;  set; }
		public double Weight { get;  set; }
		public double WeightDelta { get;  set; }
	}
}
