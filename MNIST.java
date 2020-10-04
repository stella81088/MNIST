/*
 * Stella (Wei Xing) Li
 * CWID: 102-53-174
 * Due Date: Thursday October 17 2019 at 9:00 am
 * Description: To implement and train a multi-layer (3) neural network to recognize the MNIST digit set from 0-9
 * 
 */

//import java.util.Arrays; //import array
import java.lang.Math; //import math equations
import java.io.BufferedReader; //import buffer reader
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.io.PrintStream;
import java.util.Random;
import java.util.Scanner;

public class MNIST
{
	// initialize variables
	static double [][] xvalue;
	static double [][] yvalue; 
	static double[][] weight1;
	static double[][] weight2;
	static double[] bias1;
	static double []bias2;
	static double [] z1;
	static double [] z2;
	static int numofepoch = 30;
	static int sizeofminibatch = 10;
	static int sizeofinput = 784;
	static int hiddenlayernodenum = 30;
	static double[] layer1;
	static int outputlayernodenum = 10;
	static double[] layer2;
	static double cost;
	static double[] outputerror;
	static double[][] gradiantweight2;
	static double[] layer1error;
	static double[][] gradiantweight1;
	static double[] sumoftc;
	static double[] revisedbias1;
	static int learningrate = 3;
	static double[][] sumgradiantweight1;
	static double[][] revisedweight1;
	static double[] sumoutputerror;
	static double[] revisedbias2;
	static double[][] revisedweight2;
	static double[][] sumgradiantweight2;
	static int totalinputs = 60000;
	static Random rgen = new Random();
	static int[] randomnumber;
	static double[][]xminibatch;
	static double[][]yminibatch;
	static double[] highlayer2;
	static double[] correct;
	static double[] total;
	static double totalcorrect;
	static double[] correct1;
	static double[] total1;
	static double totalcorrect1 = 0;
	static PrintStream o;
	static PrintStream console;





	public static void settingup() { //get minibatch and set up arrays

		double numofminibatch = totalinputs/sizeofminibatch; 
		
		//System.out.println (numofminibatch);
		for (int i = 0; i < numofepoch; i++) { // number of epochs it will iterate
			totalcorrect = 0;
			int[] randomnumber1 = randomgenerator(randomnumber); //generate a random number from 0-59999
			/*for (int h = 0; h < outputlayernodenum; h++) {
				correct[h] = 0;
				total[h] = 0;
			}*/


			for (int j = 0; j < numofminibatch; j++) { //number of minibatches it will iterate
				//sumoftc[j]=0;
				//makes sure the x and y values line up
				for (int a =0; a < sizeofminibatch; a++) {
					for (int b = 0; b < sizeofinput; b ++) {
						xminibatch[a][b] = xvalue[randomnumber1[10*j+a]][b];

					}
					for (int c = 0; c < outputlayernodenum; c ++) {
						yminibatch[a][c] = yvalue[randomnumber1[10*j+a]][c];
					}
				}


				for (int k =0; k < sizeofminibatch; k++) { //will iterate with number of individual inputs
					highlayer2[0] = 0; //index
					highlayer2[1] = 0; //value

					fpass(xminibatch[k],yminibatch[k]); //which xvalue array
					cost = 0; //reset for every cost
					for (int s = 0; s < outputlayernodenum; s++) {
						cost += Math.pow(layer2[s]-yminibatch[k][s],2);
					}
					cost *= 0.5;
					//System.out.println (cost);
					/*for (int w = 0; w < outputlayernodenum; w++) {
						if (highlayer2[1] < layer2[w]) {
							highlayer2[0] = w;
							highlayer2[1] = layer2[w];
						}
					}


					if (yminibatch[k][(int)highlayer2[0]] == 1) {
						correct[(int)highlayer2[0]] += 1;
					}
					total[(int)highlayer2[0]] += 1;
					 */
					bpass(yminibatch[k],xminibatch[k]); //does backpass
					//System.out.println (layer1error[k]);		
					//System.out.println();
				}

				revisedbiasandweights();
				//System.out.println();
				//System.out.println();

			} // for j

			//for (int g = 0; g < outputlayernodenum; g++) {
			//System.out.println(Arrays.toString(total));
			//System.out.println (g +" : " + correct[g] + "/" + total[g] + " = " +(correct[g]/total[g]));
			//totalcorrect += correct[g];
			//}
			//System.out.println("The accuracy is: " + (totalcorrect/60000));
			//System.out.println();
		} //for i



	} // for settingup()
	
	//generate random numbers
	public static int[] randomgenerator(int[] array){
		Random rgen = new Random();  // Random number generator/shuffle position	

		for (int i=0; i<array.length; i++) {
			int randomPosition = rgen.nextInt(array.length);
			int temp = array[i];
			array[i] = array[randomPosition];
			array[randomPosition] = temp;
		}
		//System.out.println(Arrays.toString(array));
		return array;
	}

	public static void fpass(double[] x1value, double[]ycorrect) { //forward pass

		//System.out.println (Arrays.toString(x1value)); //all arrays
		for (int i = 0; i < hiddenlayernodenum; i++) { //forward pass through layer 1
			z1[i] = 0; //intializing to set back to 0
			for (int j = 0; j < sizeofinput; j++) {
				//System.out.println ("weight =" + weight1[i][j]);
				//System.out.println ("x1value =" + x1value[j]);
				z1[i] += weight1[i][j]*x1value[j];
				//System.out.println(weight1[i][j]);

				//System.out.println ("i " + i + "j " +j);

			}
			z1[i] += bias1[i];
			//System.out.println(bias1[i]);
			//System.out.println(z1[i]);
			layer1[i] = 1/(1+(1/Math.pow(2.71828,z1[i])));
			//System.out.println("layer1 = " +layer1[i]);
		}



		for (int i = 0; i < outputlayernodenum; i++){ //forward pass through output layer
			z2[i] = 0;

			for (int j = 0; j < hiddenlayernodenum; j++) {
				z2[i] += weight2[i][j]*layer1[j];
				//System.out.println(weight2[i][j]);
			}
			z2[i] += bias2[i];
			//System.out.println (z2[i]);

			layer2[i] = 1/(1 +(1/Math.pow(2.71828,z2[i])));

			//System.out.println ("layer 2 " + layer2[i]);
			//System.out.println (i);
		}


	}

	public static void bpass(double[] y1value, double[] x1value) { //backward pass
		for (int i = 0; i <outputlayernodenum; i++) { //output layer
			outputerror[i] = (layer2[i] - y1value[i])*layer2[i]*(1-layer2[i]);
			sumoutputerror[i] += outputerror[i];
			//System.out.println (outputerror[i]);
			for (int j = 0; j < hiddenlayernodenum; j++) {
				gradiantweight2[i][j] = layer1[j]*outputerror[i];
				sumgradiantweight2[i][j] += gradiantweight2[i][j];
				//System.out.println (layer2[i]);
				//System.out.println (gradiantweight2[i][j]);
			}
		}

		for (int i = 0; i < hiddenlayernodenum; i ++) { //input layer
			layer1error[i] = 0;		
			for (int j = 0; j < outputlayernodenum; j++) {
				layer1error[i] += weight2[j][i] * outputerror[j];
			}
			layer1error[i] *= (layer1[i]*(1-layer1[i]));
			sumoftc[i] += layer1error[i];
			//System.out.println (sumoftc[i]);
			//System.out.println(layer1error[i]);

			for (int k = 0; k < sizeofinput; k++) {
				gradiantweight1[i][k] = x1value[k]*layer1error[i];
				sumgradiantweight1[i][k] += gradiantweight1[i][k];
				//System.out.println(x1value[k]);
				//System.out.println(gradiantweight1[i][k]);
			}

		}



	}
	public static void revisedbiasandweights() {
		//layer 1
		for (int i = 0; i < hiddenlayernodenum; i ++) { //getting revised bias1
			revisedbias1[i] = bias1[i]-(learningrate/2)*sumoftc[i];
			//System.out.println(sumoftc[i]);
			bias1[i] = revisedbias1[i];
			sumoftc[i] = 0;
			//System.out.println (revisedbias1[i]);
		}
		for (int i = 0; i < hiddenlayernodenum; i++) { //getting the new gradiant weight1
			for (int j = 0; j < sizeofinput; j++) {
				revisedweight1[i][j] = weight1[i][j] - (learningrate/2) * sumgradiantweight1[i][j];
				//System.out.println (revisedweight1[i][j]);
				weight1[i][j] = revisedweight1[i][j];
				//System.out.println(weight1[i][j]);
				sumgradiantweight1[i][j] = 0;
			}

			//System.out.println ();
		}
		//layer 2
		for (int i = 0; i < outputlayernodenum; i++) { //getting revised bias2
			revisedbias2[i] = bias2[i]-(learningrate/2)*sumoutputerror[i];
			//System.out.println(sumoutputerror[i]);
			bias2[i] = revisedbias2[i];
			sumoutputerror[i] = 0;
			//System.out.println (revisedbias2[i]);
		}
		for (int i = 0; i < outputlayernodenum; i++) { //getting the new gradiant weight2
			for (int j = 0; j < hiddenlayernodenum; j++) {
				revisedweight2[i][j] = weight2[i][j] - (learningrate/2) * sumgradiantweight2[i][j];
				//System.out.println (sumgradiantweight2[i][j]);
				//System.out.println (revisedweight2[i][j]);
				weight2[i][j] = revisedweight2[i][j];
				sumgradiantweight2[i][j] =0;
			}
			//System.out.println ();
		}

	}




	public static void main(String[] args) throws 
	IOException {



		// setting values to x and y
		xvalue = new double[60000][784];
		yvalue = new double[60000][10];
		/*
		// first set
		xvalue[0][0] = 0;
		xvalue[0][1] = 1;
		xvalue[0][2] = 0;
		xvalue[0][3] = 1;
		yvalue[0][0] = 0;
		yvalue[0][1] = 1;

		//second set
		xvalue[1][0] = 1;
		xvalue[1][1] = 0;
		xvalue[1][2] = 1;
		xvalue[1][3] = 0;
		yvalue[1][0] = 1;
		yvalue[1][1] = 0;

		//thrid set
		xvalue[2][0] = 0;
		xvalue[2][1] = 0;
		xvalue[2][2] = 1;
		xvalue[2][3] = 1;
		yvalue[2][0] = 0;
		yvalue[2][1] = 1;

		//fourth set
		xvalue[3][0] = 1;
		xvalue[3][1] = 1;
		xvalue[3][2] = 0;
		xvalue[3][3] = 0;
		yvalue[3][0] = 1;
		yvalue[3][1] = 0;
		 */

		//setting array limits 
		z1 = new double[hiddenlayernodenum];
		z2 = new double[outputlayernodenum];
		layer1 = new double [hiddenlayernodenum];
		layer2 = new double [outputlayernodenum];
		outputerror = new double [outputlayernodenum];
		gradiantweight2 = new double [outputlayernodenum][hiddenlayernodenum];
		layer1error = new double [hiddenlayernodenum];
		gradiantweight1 = new double [hiddenlayernodenum][sizeofinput];
		sumoftc = new double [hiddenlayernodenum];
		revisedbias1 = new double [hiddenlayernodenum];
		sumgradiantweight1 = new double[hiddenlayernodenum][sizeofinput];
		revisedweight1 = new double[hiddenlayernodenum][sizeofinput];
		sumoutputerror = new double[outputlayernodenum];
		revisedbias2 = new double [outputlayernodenum];
		revisedweight2 = new double [outputlayernodenum][hiddenlayernodenum];
		sumgradiantweight2 = new double[outputlayernodenum][hiddenlayernodenum];
		weight1 = new double[hiddenlayernodenum][sizeofinput];
		weight2 = new double[outputlayernodenum][hiddenlayernodenum];
		bias1 = new double[hiddenlayernodenum];
		bias2 = new double[outputlayernodenum];
		randomnumber = new int [totalinputs];
		xminibatch = new double [10][sizeofinput];
		yminibatch = new double [10][outputlayernodenum];
		highlayer2 = new double[2];
		correct = new double [outputlayernodenum];
		total = new double[outputlayernodenum];
		correct1 = new double [outputlayernodenum];
		total1 = new double[outputlayernodenum];
		
		
		for (int i = 0; i < totalinputs; i++) {
			randomnumber[i] = i;
		}
		Random r = new Random();
		//randomizing weight1 and bias1
		for (int i = 0; i < hiddenlayernodenum; i++) {
			bias1[i] = -1 + (1 +1) * r.nextDouble();
			//System.out.println(bias1[i]);
			for (int j = 0; j < sizeofinput; j++) {
				weight1[i][j] = -1 + (1 +1) * r.nextDouble();
				//System.out.println(weight1[i][j]);
			}
		}

		//randomizing weight2 and bias2
		for (int i = 0; i < outputlayernodenum; i++) {
			bias2[i] = -1 + (1 +1) * r.nextDouble();
			//System.out.println(bias2[i]);
			for (int j = 0; j < hiddenlayernodenum; j++) {
				weight2[i][j] = -1 + (1 +1) * r.nextDouble();
				//System.out.println(weight2[i][j]);
			}
		}
		boolean f = true;
		while (f == true) {
			//output statements
			System.out.println("Hello! Welcome to the MNIST handwritten digit recognizer.");
			System.out.println("Press 1 to train the data");
			System.out.println("Press 2 to load the saved data from the file (refer to option #5");
			System.out.println("Press 3 to display the training set accuracy");
			System.out.println("Press 4 to display the testing set accuracy");
			System.out.println("Press 5 to to save network state to a file");
			System.out.println("Press 6 or anything else to exit the program");
			
			//Scanner to recognize user input
			Scanner scan = new Scanner(System.in);
			int answer = scan.nextInt();
			
			//If user presses 1
			if (answer == 1) {
				//reading training csv file
				BufferedReader trainingcsv = null;
				try {
					String currentline;

					trainingcsv = new BufferedReader(new FileReader("/Users/Stella_xo/Desktop/mnist_train.csv")); //find file
					int linecounter = 0; //sets linecounter

					while ((currentline = trainingcsv.readLine()) != null) {	//print out the lines
						String[] currentline1 = currentline.split(",");
						//setting up x values
						for (int i = 1; i < sizeofinput ; i++) {
							xvalue[linecounter][i-1] = Double.valueOf(currentline1[i])/255.0; //normalize and rbg value
							// System.out.println(Arrays.toString(xvalue[linecounter]));

						}
						yvalue[linecounter][Integer.valueOf(currentline1[0])] = 1; //turns yvalue into 'one hot' vector
						//System.out.println(Arrays.toString(yvalue[linecounter]));
						linecounter += 1;
						//System.out.println(currentline);
					}

				} catch (IOException e) { //catch exception if program cannot create/write to the file

					e.printStackTrace(); //print out what error and where in the code

				} finally {	//will still execute even if there is an error

					try {
						if (trainingcsv != null)
							trainingcsv.close();	//close the file
					} catch (IOException ex) {
						ex.printStackTrace();

					}
				}

				settingup(); //call upon settingup function where the training happens
			
			//if user pressed 2
			}else if (answer ==2) {
				
				
				BufferedReader csv = null;
				try {
					String currentline2;

					csv = new BufferedReader(new FileReader("/Users/Stella_xo/eclipse-workspace/MNIST/data.txt")); //find file
					//int linecounter = 0; //sets linecounter

					while ((currentline2 = csv.readLine()) != null) {	//print out the lines
						int counter =0;
						String[] currentline1 = currentline2.split(",");

						//change bias1 to the bias 1 found saved in file
						for (int i = 0; i < hiddenlayernodenum; i++) {
							System.out.println(counter);
							bias1[i] = Double.valueOf(currentline1[counter]);
							counter ++; 
						}
						
						//change weight1 to the weight 1 found saved in file
						for (int i = 0; i < hiddenlayernodenum; i++) {
							for (int j = 0; j < sizeofinput; j++) {
								weight1[i][j] = Double.valueOf(currentline1[counter]);
								counter ++; 
							}
						}
						
						//change bias2 to the bias 2 found saved in file
						for (int i = 0; i < outputlayernodenum; i++) {
							bias2[i] = Double.valueOf(currentline1[counter]);
							counter ++; 
						}
						
						//change weight2 to the weight 2 found saved in file
						for (int i = 0; i < outputlayernodenum; i++) {
							for (int j = 0; j < hiddenlayernodenum; j++) {
								weight2[i][j] = Double.valueOf(currentline1[counter]);
								counter ++; 
							}
						}
					}

				} catch (IOException e) { //catch exception if program cannot create/write to the file

					e.printStackTrace(); //print out what error and where in the code

				} finally {	//will still execute even if there is an error

					try {
						if (csv != null)
							csv.close();	//close the file
					} catch (IOException ex) {
						ex.printStackTrace();

					}
				}

			//if user presses 3
			}else if (answer == 3) {
				double numofminibatch = totalinputs/sizeofminibatch; 
				for (int i = 0; i < numofepoch; i++) { // number of epochs it will iterate
					totalcorrect = 0;
					int[] randomnumber1 = randomgenerator(randomnumber);
					for (int h = 0; h < outputlayernodenum; h++) {
						correct[h] = 0;
						total[h] = 0;
					}


					for (int j = 0; j < numofminibatch; j++) { //number of minibatches it will iterate
						//sumoftc[j]=0;
						for (int a =0; a < sizeofminibatch; a++) {
							for (int b = 0; b < sizeofinput; b ++) {
								xminibatch[a][b] = xvalue[randomnumber1[10*j+a]][b];

							}
							for (int c = 0; c < outputlayernodenum; c ++) {
								yminibatch[a][c] = yvalue[randomnumber1[10*j+a]][c];
							}
						}


						for (int k =0; k < sizeofminibatch; k++) { //will iterate with number of individual inputs
							highlayer2[0] = 0; //index
							highlayer2[1] = 0; //value

							fpass(xminibatch[k],yminibatch[k]); //which xvalue array
							cost = 0; //reset for every cost
							for (int s = 0; s < outputlayernodenum; s++) {
								cost += Math.pow(layer2[s]-yminibatch[k][s],2);
							}
							cost *= 0.5;
							//System.out.println (cost);
							for (int w = 0; w < outputlayernodenum; w++) { //finding highest layer2 value out of 10
								if (highlayer2[1] < layer2[w]) {
									highlayer2[0] = w;
									highlayer2[1] = layer2[w];
								}
							}


							if (yminibatch[k][(int)highlayer2[0]] == 1) { //find the # of correct for each output
								//System.out.println(highlayer2[0]);
								correct[(int)highlayer2[0]] += 1;
								//System.out.println(correct[(int)highlayer2[0]]);
							}
							total[(int)highlayer2[0]] += 1; //total number of iterations
							//System.out.println(total[(int)highlayer2[0]]);
							bpass(yminibatch[k],xminibatch[k]); 
							//System.out.println (layer1error[k]);		
							//System.out.println();
						}

						revisedbiasandweights();
						//System.out.println();
						//System.out.println();

					} // for j

					for (int g = 0; g < outputlayernodenum; g++) {
						//System.out.println(Arrays.toString(total));
						//after each epoch for each output
						System.out.println (g +" : " + correct[g] + "/" + total[g] + " = " +((correct[g]/total[g])*100) + "%" );
						totalcorrect += correct[g];
					}
					//after each epoch
					System.out.println("The accuracy is: " + ((totalcorrect/60000)*100) + "%");
					System.out.println();
				} //for i
				
			//if user presses 4
			}else if (answer ==4) {
				//int totalinputs = 10000;

				BufferedReader trainingcsv = null;
				try {
					String currentline;

					trainingcsv = new BufferedReader(new FileReader("/Users/Stella_xo/Desktop/mnist_test.csv")); //find file
					int linecounter = 0; //sets linecounter

					while ((currentline = trainingcsv.readLine()) != null) {	//print out the lines
						String[] currentline1 = currentline.split(",");
						//setting up x values
						for (int i = 1; i < sizeofinput ; i++) {
							xvalue[linecounter][i-1] = Double.valueOf(currentline1[i])/255.0; //normalize and rbg value
							//  System.out.println(Arrays.toString(xvalue[linecounter]));
							// System.out.println();	  
						}
						yvalue[linecounter][Integer.valueOf(currentline1[0])] = 1; //turns yvalue into 'one hot' vector
						//System.out.println(Arrays.toString(yvalue[linecounter]));
						linecounter += 1;
						//System.out.println(currentline);
						// System.out.println();
					}

				} catch (IOException e) { //catch exception if program cannot create/write to the file

					e.printStackTrace(); //print out what error and where in the code

				} finally {	//will still execute even if there is an error

					try {
						if (trainingcsv != null)
							trainingcsv.close();	//close the file
					} catch (IOException ex) {
						ex.printStackTrace();

					}
				}
				
				//System.out.println("Read");
				
				
				for (int i = 0; i < 10000; i++) { // number of epochs it will iterate
					
					fpass(xvalue[i],yvalue[i]); //which xvalue array
				}
				for (int g = 0; g < outputlayernodenum; g++) {
					//System.out.println(Arrays.toString(total));
					//after each epoch for each output
					System.out.println (g +" : " + correct[g] + "/" + total[g] + " = " +((correct[g]/total[g])*100) + "%" );
					totalcorrect += correct[g];
				}
				//after each epoch
				System.out.println("The accuracy is: " + ((totalcorrect/60000)*100) + "%");
				System.out.println();
					


					
					

			//if user presses 5
			}else if (answer == 5) {
				//output to new file
				o = new PrintStream("data.txt");
				console = System.out;
				System.setOut(o);

				for (int i = 0; i < hiddenlayernodenum; i++) {
					System.out.print(bias1[i]+",");
				}

				for (int i = 0; i < hiddenlayernodenum; i++) {
					for (int j = 0; j < sizeofinput; j++) {
						System.out.print(weight1[i][j]+",");
					}
				}
				//weight2 and bias2
				for (int i = 0; i < outputlayernodenum; i++) {
					System.out.print(bias2[i]+",");
				}
				for (int i = 0; i < outputlayernodenum; i++) {
					for (int j = 0; j < hiddenlayernodenum; j++) {
						System.out.print(weight2[i][j]+",");
					}
				}
				//set back so user can input more things
				System.setOut(console);
				
			//if user presses anything else
			}else {
				f = false; // while loop will end
			}
		} 
	} //public static void main
} //public MNIST