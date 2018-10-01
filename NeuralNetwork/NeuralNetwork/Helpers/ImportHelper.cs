using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.IO;
using System.Linq;
using NeuralNetwork.NetworkModels;
using Newtonsoft.Json;

namespace NeuralNetwork.Helpers
{
	public static class ImportHelper
	{
		public static Network ImportNetwork()
		{
            var allHelperNeurons = new Dictionary<Guid, Neuron>();

            var dn = GetHelperNetwork();

            if (dn == null) return null;

            NeuralLayer inputLayer = new NeuralLayer();
            List<NeuralLayer> hiddenLayer = new List<NeuralLayer>();
            NeuralLayer outputLayer = new NeuralLayer();

            //Input Layer
            foreach (var n in dn.InputLayer)
			{
                IActivationFunction activationFunction = null;
                if(n.ActivationFunctionType!=null)
                    activationFunction = (IActivationFunction) Activator.CreateInstance(Type.GetType(n.ActivationFunctionType));

				var neuron = new Neuron(activationFunction, n.Id,n.Bias,n.BiasDelta,n.Gradient,n.Value);
                inputLayer.Add(neuron);
                allHelperNeurons.Add(n.Id,neuron);
            }

			//Hidden Layers
			foreach (var layer in dn.HiddenLayers)
			{
                var enumNeuron = layer.Select(n => {

                    IActivationFunction activationFunction = null;
                    if (n.ActivationFunctionType != null)
                        activationFunction = (IActivationFunction)Activator.CreateInstance(Type.GetType(n.ActivationFunctionType));
                    var neuron = new Neuron(activationFunction, n.Id, n.Bias, n.BiasDelta, n.Gradient, n.Value);
                    allHelperNeurons.Add(n.Id, neuron);
                    return neuron;
                });

                var neurons = new NeuralLayer(enumNeuron);
                hiddenLayer.Add(neurons);
            }
            
            //Export Layer
            foreach (var n in dn.OutputLayer)
			{
                IActivationFunction activationFunction = null;
                if (n.ActivationFunctionType != null)
                    activationFunction = (IActivationFunction)Activator.CreateInstance(Type.GetType(n.ActivationFunctionType));
                var neuron = new Neuron(activationFunction, n.Id, n.Bias, n.BiasDelta, n.Gradient, n.Value);
                outputLayer.Add(neuron);
                allHelperNeurons.Add(n.Id, neuron);
            }

            //Synapses
            foreach (var syn in dn.Synapses)
			{
                var inputNeuron = allHelperNeurons[syn.InputNeuronId];
				var outputNeuron = allHelperNeurons[syn.OutputNeuronId];
                var synapse = new Synapse(syn.Id, inputNeuron, outputNeuron, syn.Weight, syn.WeightDelta );
				inputNeuron.AddOutputSynapse(synapse);
				outputNeuron.AddInputSynapse(synapse);
			}

            return new Network(dn.LearnRate, dn.Momentum, inputLayer, hiddenLayer, outputLayer);
		}

		public static List<DataSet> ImportDatasets()
		{
			try
			{
				var dialog = new OpenFileDialog
				{
					Multiselect = false,
					Title = "Open Dataset File",
					Filter = "Text File|*.txt;"
				};

				using (dialog)
				{
					if (dialog.ShowDialog() != DialogResult.OK) return null;
					using (var file = File.OpenText(dialog.FileName))
					{
						return JsonConvert.DeserializeObject<List<DataSet>>(file.ReadToEnd());
					}
				}
			}
			catch (Exception)
			{
				return null;
			}
		}

		private static HelperNetwork GetHelperNetwork()
		{
			try
			{
				var dialog = new OpenFileDialog
				{
					Multiselect = false,
					Title = "Open Network File",
					Filter = "Text File|*.txt;"
				};

				using (dialog)
				{
					if (dialog.ShowDialog() != DialogResult.OK) return null;

					using (var file = File.OpenText(dialog.FileName))
					{
						return JsonConvert.DeserializeObject<HelperNetwork>(file.ReadToEnd());
					}
				}
			}
			catch (Exception)
			{
				return null;
			}
		}
	}
}
