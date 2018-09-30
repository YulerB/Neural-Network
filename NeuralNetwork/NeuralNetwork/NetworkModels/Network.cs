using NeuralNetwork.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NeuralNetwork.NetworkModels
{
	public class Network
	{
        private double LearnRate;
        private double Momentum;
        private NeuralLayer InputLayer;
        private List<NeuralLayer> HiddenLayers;
        private NeuralLayer OutputLayer;

		private static readonly Random Random = new Random();

        public Network(double learnRate, double momentum, NeuralLayer inputLayer, List<NeuralLayer> hiddenLayers, NeuralLayer outputLayer)
        {
            LearnRate = learnRate;
            Momentum = momentum;
            InputLayer = inputLayer;
            HiddenLayers = hiddenLayers;
            OutputLayer = outputLayer;
        }

        public Network(int inputSize, int[] hiddenSizes, int outputSize, double? learnRate = null, double? momentum = null)
		{
			LearnRate = learnRate ?? .4;
			Momentum = momentum ?? .9;
			HiddenLayers = new List<NeuralLayer>(hiddenSizes.Length);
            InputLayer = new NeuralLayer(inputSize);
            HiddenLayers.Add(new NeuralLayer(hiddenSizes[0], InputLayer));
            HiddenLayers.AddRange(Enumerable.Range(1, hiddenSizes.Length - 1).Select(i => new NeuralLayer(Enumerable.Range(0, hiddenSizes[i]).Select(_ => new Neuron(HiddenLayers[i - 1])))));
            OutputLayer = new NeuralLayer(outputSize, HiddenLayers.Last());
		}

        public int HiddenLayersCount
        {
            get
            {
                return HiddenLayers.Count;
            }
        }
        public int HiddenLayerNeuronCount
        {
            get
            {
                return HiddenLayers.Sum(_=> _.Count);
            }
        }

        public void Train(List<DataSet> dataSets, int numEpochs)
		{
			for (var i = 0; i < numEpochs; i++)
			{
                dataSets.ForEach(dataSet =>{ ForwardPropagate(dataSet.Values); BackPropagate(dataSet.Targets); });
			}
		}

		public int Train(List<DataSet> dataSets, double minimumError)
		{
			var numEpochs = 0;
            while (
                dataSets
                    .Select(_ =>
                    {
                        ForwardPropagate(_.Values);
                        BackPropagate(_.Targets);
                        return CalculateError(_.Targets);
                    })
                    .Average() > minimumError 
                && 
                ++numEpochs < int.MaxValue
            );
            return numEpochs;
		}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ForwardPropagate(params double[] inputs)
		{
            InputLayer.SetInputs(inputs);
			HiddenLayers.ForEach(a => a.CalculateValues());
			OutputLayer.ForEach(a => a.CalculateValue());
		}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void BackPropagate(params double[] targets)
		{
            OutputLayer.CalculateGradients(targets);
            OutputLayer.UpdateBias(LearnRate, Momentum);
            for (var j = HiddenLayers.Count - 1; j > -1; j--) HiddenLayers[j].CalculateGradients();
            for (var j = HiddenLayers.Count - 1; j > -1; j--) HiddenLayers[j].UpdateBias(LearnRate, Momentum);
		}

		public double[] Compute(params double[] inputs)
		{
			ForwardPropagate(inputs);
			return OutputLayer.GetValues().ToArray();
		}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double CalculateError(params double[] targets)
		{
			var i = 0;
            return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
        }

        public static double GetRandom()
		{
			return 2 * Random.NextDouble() - 1;
		}

        public HelperNetwork ToHelperNetwork()
        {
            List<HelperNeuron> hnInputLayer = new List<HelperNeuron>();
            List<HelperSynapse> hnSynapses = new List<HelperSynapse>();
            List<List<HelperNeuron>> hnHiddenLayers = new List<List<HelperNeuron>>();
            List<HelperNeuron> hnOutputLayer = new List<HelperNeuron>();
            //Input Layer
            foreach (var n in InputLayer)
            {
                hnInputLayer.Add(n.ToHelperNeuron());
                hnSynapses.AddRange(n.ToOutputHelperSynapses());
            }

            //Hidden Layer
            foreach (var l in HiddenLayers)
            {
                var layer = new List<HelperNeuron>();

                foreach (var n in l)
                {
                    layer.Add(n.ToHelperNeuron());
                    hnSynapses.AddRange(n.ToOutputHelperSynapses());
                }

                hnHiddenLayers.Add(layer);
            }

            //Output Layer
            foreach (var n in OutputLayer)
            {
                hnOutputLayer.Add(n.ToHelperNeuron());
                hnSynapses.AddRange(n.ToOutputHelperSynapses());
            }

            return new HelperNetwork(this.LearnRate, this.Momentum, hnInputLayer , hnHiddenLayers, hnOutputLayer, hnSynapses);            
        }
	}

	#region -- Enum --
	public enum TrainingType
	{
		Epoch,
		MinimumError
	}
	#endregion
}