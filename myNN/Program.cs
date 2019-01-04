using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
//Mail Systems
using System.Net;
using System.Net.Mail;

namespace myNN
{
    class MainClass
    {
        //public static bool sendEmail = true;
        public static bool sendEmail = false;

        public static void Main(string[] args)
        {
            //List<double> testData = new List<double> { 1, -1, 0, 1, -1, 1 };
            //Console.WriteLine(testData.normalize(-1,1).ToArray().arrayToString());
            //;
            //Maybe neuron delta derivative can be calc by input.sum and take deriviative?
            //Make training data equal, too many 0s

            NN.neuralNet mynn = new NN.neuralNet();

            mynn.addLayer(2, 2, 2, NN.activationFunction.input); //Input
            //mynn.addLayer(2, 2, 2, NN.activationFunction.tanh, true); //Hidden
            mynn.addLayer(2, 2, 2, NN.activationFunction.tanh); //Hidden
            //mynn.addLayer(2, 2, 2, NN.activationFunction.tanh); //Hidden
            mynn.addLayer(1, 2, 1, NN.activationFunction.tanh);//, true); //Output

            //mynn.addLayer(2, 2, 3, NN.activationFunction.input); //Input
            //mynn.addLayer(3, 2, 3, NN.activationFunction.sigmoid);
            //mynn.addLayer(3, 3, 3, NN.activationFunction.sigmoid);
            //mynn.addLayer(1, 3, 1, NN.activationFunction.sigmoid);

            mynn.setLearn(NN.learningMode.staticLearn, .5);


            List<double[]> ins = new List<double[]>();
            List<double[]> outs = new List<double[]>();

            //ins.Add(new double[] { 0, 0 }); outs.Add(new double[] { 0 });
            //ins.Add(new double[] { 0, 1 }); outs.Add(new double[] { 1 });
            //ins.Add(new double[] { 1, 0 }); outs.Add(new double[] { 1 });
            //ins.Add(new double[] { 1, 1 }); outs.Add(new double[] { 1 });

            //ins.Add(new double[] { 0, 0 }); outs.Add(new double[] { 0 });
            //ins.Add(new double[] { 0, 1 }); outs.Add(new double[] { 0 });
            //ins.Add(new double[] { 1, 0 }); outs.Add(new double[] { 0 });
            //ins.Add(new double[] { 1, 1 }); outs.Add(new double[] { 1 });

            ins.Add(new double[] { 0, 0 }); outs.Add(new double[] { 0 });
            ins.Add(new double[] { 0, 1 }); outs.Add(new double[] { 1 });
            ins.Add(new double[] { 1, 0 }); outs.Add(new double[] { 1 });
            ins.Add(new double[] { 1, 1 }); outs.Add(new double[] { 0 });

            //Console.WriteLine(new double[] { 6.5,5,3 }.ToList<double>().normalize().ToArray().arrayToString());
            //return;

            int c = 0;
            while (true)
            {
                c++;
                if (c >= 5000)
                {
                    c = 0;
                    mynn.backward(ins, outs, false);
                    double error = mynn.getError(ins, outs);
                    Console.WriteLine("Total Error: " + error);
                    Console.WriteLine("Net Momentum: " + mynn.getNetMomentum());
                    logFail(ins, outs, mynn);
                }

                if (good(ins, outs, mynn))
                {
                    Console.WriteLine("Converged!");
                    break;
                }
            }

            Console.WriteLine("Weight Matrix: ");
            for (int i = 1; i < mynn.layers.Count; i++)
            {
                Console.Write("[{0}]", i);
                for (int j = 0; j < mynn.layers[i].neurons.Length; j++)
                {
                    Console.Write(mynn.layers[i].neurons[j].weights.arrayToString() + ",");
                }
                Console.WriteLine();
            }
            Console.WriteLine("{Network Out}, {Expected Out}: ");
            logOuts(ins, outs, mynn);
            //if (sendEmail) mailDone("Converged");
            //mynn.save("nn");
            Console.WriteLine(mynn.toMath());
            //Console.WriteLine(mynn.solve(new double[] { 0 }, new double[] { 0 }, mynn.getError(ins, outs), 0, 1));
            //Console.ReadLine();
        }

        public static bool good(List<double[]> ins, List<double[]> outs, NN.neuralNet myNN)
        {
            for (int i = 0; i < ins.Count; i++)
            {
                double[] output = myNN.forward(ins[i]);
                //Console.WriteLine(output.arrayToString()+","+outs[i].arrayToString());
                for (int j = 0; j < output.Length; j++)
                {
                    //Console.WriteLine(((int)Math.Round(output[j])) + "," + (int)Math.Round(outs[i][j]));
                    //if ((((int)Math.Round(output[j], 1))) != ((int)Math.Round(outs[i][j]))) return false;
                    //if ((((int)Math.Round(output[j]))) != ((int)Math.Round(outs[i][j]))) return false;
                    if (!binaryCompare(output[j], outs[i][j], .08)) return false;
                }
            }
            return true;
        }

        public static void logFail(List<double[]> ins, List<double[]> outs, NN.neuralNet myNN)
        {
			for (int i = 0; i < ins.Count; i++)
			{
				double[] output = myNN.forward(ins[i]);

				for (int j = 0; j < output.Length; j++)
				{
                    //if (((int)Math.Round(output[j],1)) != (int)Math.Round(outs[i][j])) { Console.WriteLine(output.arrayToString() + "," + outs[i].arrayToString()); return; }
                    //if (((int)Math.Round(output[j])) != (int)Math.Round(outs[i][j])) { Console.WriteLine(output.arrayToString() + "," + outs[i].arrayToString()); return; }
                    if (!binaryCompare(output[j], outs[i][j], .08)) { Console.WriteLine(output.arrayToString() + "," + outs[i].arrayToString()); return; }
				}
			}
        }

        public static void logOuts(List<double[]> ins, List<double[]> outs, NN.neuralNet myNN)
        {
            for (int i = 0; i < ins.Count; i++)
            {
                double[] output = myNN.forward(ins[i]);

                for (int j = 0; j < output.Length; j++)
                {
                    Console.WriteLine(output.arrayToString() + "," + outs[i].arrayToString());
                }
            }
        }

        public static bool binaryCompare(double x, double expected, double threshold)
        {
            double activate = 1 - threshold;
            if ((x > activate) && (expected > activate)) return true;
            if ((x < threshold) && (expected < threshold)) return true;
            return false;
        }

        public static void mailDone(string body)
        {

            var client = new SmtpClient("smtp.gmail.com", 587)
            {
                Credentials = new NetworkCredential("maximkha.notification@gmail.com", "kukunist"),
                EnableSsl = true
            };
            //client.Send("myusername@gmail.com", "myusername@gmail.com", "test", "testbody");
            client.Send("maximkha.notification@gmail.com", "maximkhanov.ku@icloud.com", "AI notification", body);
            Console.WriteLine("Sent");
        }
    }

    public static class Extensions
    {
        public static string arrayToString(this Array array)
        {
            string s = "{";
            bool f = true;
            foreach (var item in array)
            {
                if (f) { f = false; s += item.ToString(); continue; }
                s += ",";
                s += item.ToString();
            }
            s += "}";
            return s;
        }
	}
}
