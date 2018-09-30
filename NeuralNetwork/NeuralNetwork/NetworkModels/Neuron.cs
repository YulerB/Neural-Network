using NeuralNetwork.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NeuralNetwork.NetworkModels
{

    public class Neuron
	{
        private Guid Id;
        private Synapses InputSynapses;
        private Synapses OutputSynapses;
        private double Bias;
        private double BiasDelta;
        private double Gradient;
        private double Value;

        public EventHandler<NeuronBiasUpdatedEventArg> OnBiasUpdated;

        //private readonly IActivationFunction activationFunction = new Sigmoid();
        private readonly IActivationFunction activationFunction = new SoftPlus();

        public Neuron()
		{
			Id = Guid.NewGuid();
			InputSynapses = new Synapses();
			OutputSynapses = new Synapses();
			Bias = Network.GetRandom();
		}
        public Neuron(Guid id, double bias, double biasDelta, double gradient, double value) : this()
        {
            this.Id = id;
            this.Bias = bias;
            this.BiasDelta = biasDelta;
            this.Gradient = gradient;
            this.Value = value;
        }
		public Neuron(NeuralLayer inputNeurons) : this()
		{
            InputSynapses.AddRange(inputNeurons.Select(inputNeuron =>
            {
                var synapse = new Synapse(inputNeuron, this);
                inputNeuron.OutputSynapses.Add(synapse);
                return synapse;
            }));
            InputSynapses.ForEach(_ => this.OnBiasUpdated += _.NeuronBiasUpdated); // Receive Weight Updates for Back Propogation.
        }

        internal void SetInput(double v)
        {
            Value = v;
        }
        public double GetValue()
        {
            return Value;
        }
        public double CalculateValue()
		{
			return Value = activationFunction.Activate(InputSynapses.CalculateValue() + Bias);
		}
        public double CalculateValueByWeight(double weight)
        {
            return weight * Value;
        }        
        public double CalculateError(double target)
        {
            return target - Value;
        }
        public double CalculateGradient(double? target = null)
		{
            return target == null ? 
                Gradient = OutputSynapses.Sum(a => a.CalculateGradient()) * activationFunction.Derivative(Value) :
                Gradient = CalculateError(target.Value) * activationFunction.Derivative(Value);
        }
        public double CalculateWeightDelta (double learnRate, double gradient)
        {
            return learnRate * gradient * Value;
        }
        public double CalculateGradientByWeight(double weight)
        {
            return weight * Gradient;
        }
        public void UpdateBias(double learnRate, double momentum)
        {
            UpdateInternalBias(learnRate, momentum);
            if (OnBiasUpdated != null) this.OnBiasUpdated(this, new NeuronBiasUpdatedEventArg (learnRate, momentum, Gradient));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateInternalBias(double learnRate, double momentum)
        {
            var prevDelta = BiasDelta;
            BiasDelta = learnRate * Gradient;
            Bias += BiasDelta + momentum * prevDelta;
        }        
        public HelperNeuron ToHelperNeuron()
        {
            return new HelperNeuron(Bias,BiasDelta,Gradient,Id,Value);
        }
        public IEnumerable<HelperSynapse> ToOutputHelperSynapses()
        {
            return OutputSynapses.ToHelperSynapses();
        }
        public void AddOutputSynapse(Synapse synapse)
        {
            OutputSynapses.Add(synapse);
        }
        public void AddInputSynapse(Synapse synapse)
        {
            InputSynapses.Add(synapse);
        }
    }
}