using System.IO;
using System.Windows.Forms;
using Newtonsoft.Json;
using System.Collections.Generic;
using NeuralNetwork.NetworkModels;

namespace NeuralNetwork.Helpers
{
	public static class ExportHelper
	{
		public static void ExportNetwork(Network network)
		{
			var dn = network.ToHelperNetwork();

			var dialog = new SaveFileDialog
			{
				Title = "Save Network File",
				Filter = "Text File|*.txt;"
			};

			using (dialog)
			{
				if (dialog.ShowDialog() != DialogResult.OK) return;
				using (var file = File.CreateText(dialog.FileName))
				{
					var serializer = new JsonSerializer { Formatting = Formatting.Indented };
					serializer.Serialize(file, dn);
				}
			}
		}

		public static void ExportDatasets(List<DataSet> datasets)
		{
			var dialog = new SaveFileDialog
			{
				Title = "Save Dataset File",
				Filter = "Text File|*.txt;"
			};

			using (dialog)
			{
				if (dialog.ShowDialog() != DialogResult.OK) return;
				using (var file = File.CreateText(dialog.FileName))
				{
					var serializer = new JsonSerializer { Formatting = Formatting.Indented };
					serializer.Serialize(file, datasets);
				}
			}
		}

        //private static HelperNetwork GetHelperNetwork(Network network)
        //{
        //    return network.ToHelperNetwork();
        //}
	}
}