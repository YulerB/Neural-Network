using System;

namespace NeuralNetwork.Helpers
{
    public class HelperSynapse
	{
        public HelperSynapse(Guid id, Guid inputNeuronId, Guid outputNeuronId, double weight, double weightDelta)
        {
            this.Id = id;
            this.InputNeuronId = inputNeuronId;
            this.OutputNeuronId = outputNeuronId;
            this.Weight = weight;
            this.WeightDelta = weightDelta;
        }

        public Guid Id { get; private set; }
		public Guid OutputNeuronId { get; private set; }
		public Guid InputNeuronId { get; private set; }
		public double Weight { get; private set; }
		public double WeightDelta { get; private set; }
	}
}
