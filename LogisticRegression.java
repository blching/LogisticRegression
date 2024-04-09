import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;

/* Comment out imports for graphing chart */
/* 
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import javax.swing.*;
import java.awt.*; 
*/

/* 

LogisticRegression - Brandon Ching
ID: 01573098
brandon.l.ching@sjsu.edu

*/

public class LogisticRegression  {
        
        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the number of iterations */
        private int ITERATIONS = 200;

        //Arrays to graph logloss
        ArrayList<Double> x = new ArrayList<Double>();
        ArrayList<Double> y = new ArrayList<Double>();

        /* Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        LogisticRegression(int size) {
            weights = new double[size]; //Automatically sets to 0
        }

        /* Implementation of the sigmoid function **/
        private double sigmoid(double z) {
            return (1/(1 + Math.exp(-z)));
        }

        /* Helper function for prediction **/
        /** Takes a test instance as input and outputs the probability of the label being 1 **/
        private double predictHelper(int[] instance) {
            double z = 0;

            //Sum of instance values  * weight
            for (int i = 0; i<weights.length; i++) {
                z += instance[i] * weights[i];
            }

            //Return sum imputted into sigmoid
            return sigmoid(z);
        }


        /* The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        private int prediction(int[] instance) {
            double help = predictHelper(instance);

            //If % of being positive >= 1, predict 1
            if (help >= 0.5) return 1;

            //Else predict 0
            return 0;
        }    


        /** This function takes a test set as input, call the predict function to predict a label for it, **/
        /** and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix **/
        private void test(ArrayList<int[]> testSet) {
            int TP = 0, FP = 0, FN = 0, TN = 0; //Set True Positive, False Positive, False Negative, True Negative
            for (int i = 0; i < testSet.size(); i++) {
                int pre = prediction(testSet.get(i));
                int label = (int) testSet.get(i)[testSet.get(i).length-1];
                
                //Compares predicted label vs. real label, and counts it
                if (label == 1 && pre == 1) TP++;
                if (label == 0 && pre == 0) TN++;
                if (label == 1 && pre == 0) FN++;
                if (label == 0 && pre == 1) FP++;
                
            }

            double accuracy = (double)(TP+TN)/(TP+TN+FP+FN);

            double precisionP = (double)(TP)/(TP+FP);
            double precisionN = (double)(TN)/(TN+FN);

            double recallP = (double)(TP)/(TP+FN);
            double recallN = (double)(TN)/(TN+FP);

            double F1P = (double)(2*recallP*precisionP)/(recallP+precisionP);
            double F1N = (double)(2*recallN*precisionN)/(recallN+precisionN);


            //Print out stats for all, both positive and negative
            System.out.println("");
            System.out.println("Overall Accuracy:");
            System.out.println("Overall accuracy of Set: " + accuracy);

            System.out.println("");

            System.out.println("Positive Stats: (Spam)");
            System.out.println("Spam (Positive) Precision of Set: " + precisionP);
            System.out.println("Spam (Positive) Recall of Set: " + recallP);
            System.out.println("Spam (Positive) F1 of Set: " + F1P);

            System.out.println("");

            System.out.println("Negative Stats: (Ham)");
            System.out.println("Ham (Negative) Precision of Set: " + precisionN);
            System.out.println("Ham (Negative) Recall of Set: " + recallN);
            System.out.println("Ham (Negative) F1 of Set: " + F1N);

            System.out.println("");

            System.out.println("Confusion Matrix:");
            System.out.println("    (TP)    (FP)");
            System.out.println("    (FN)    (TN)");

            System.out.println("      Spam   Ham");
            System.out.println("Spam   " + TP + "   " + FP);
            System.out.println("Ham    " + FN + "   " + TN);

        }



        /** Train the Logistic Regression in a function using Stochastic Gradient Descent **/
        /** Also compute the log-oss in this function **/
        private void train(ArrayList<int[]> trainSet) {
            for (int j = 0; j < ITERATIONS; j++) {
                double logLoss = 0;
                for (int i = 0; i < trainSet.size(); i++) {
                    int[] current = trainSet.get(i);
                    int pre = prediction(current);
                    int label = (int) current[current.length-1];
                    double prob = predictHelper(current);

                    for (int k = 0; k < weights.length; k++) {
                        weights[k] -= (prob-label) * rate * current[k];
                    }

                    logLoss += -(label*Math.log(prob)+(1-label)*Math.log(1-prob));
                }

                logLoss = logLoss/trainSet.size();

                //System.out.println("Logloss of Iteration " + j + ": " + logLoss);
                x.add((double) j);
                y.add(logLoss);

                //Utilized for graphinh LogLoss
                //logLossSeries.add(j, logLoss);
            }
        }


        /** Function to read the input dataset 
         * @throws IOException **/
        private static ArrayList<int[]> inputFile(FileReader inFile) throws IOException {
            ArrayList<int[]> list = new ArrayList<int[]>();

            try (BufferedReader buff = new BufferedReader(inFile)) {
                buff.readLine(); //Skips first line
                String line;
                while ((line = buff.readLine()) != null) {
                    String[] tempStr = line.split(",");

                    int[] tempInt = Arrays.stream(tempStr).mapToInt(Integer::parseInt).toArray();

                    list.add(tempInt);
                } 
            }

            return list;
        }
        
        /* Function and variables to graph LogLoss */
        /* 
        private static XYSeries logLossSeries;

        private void plotLogLoss() {
                XYSeriesCollection dataset = new XYSeriesCollection();
                dataset.addSeries(logLossSeries);
    
                JFreeChart chart = ChartFactory.createXYLineChart(
                        "Log Loss vs Iterations",
                        "Iterations",
                        "Log Loss", 
                        dataset, PlotOrientation.VERTICAL, true, true, false
                );
    
                ChartPanel chartPanel = new ChartPanel(chart);
                chartPanel.setPreferredSize(new Dimension(800, 600));
    
                JFrame frame = new JFrame("Log Loss Plot");

                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(chartPanel);
                frame.pack();
                frame.setVisible(true);
            ;
        }

        */

        /** main Function 
         * @throws IOException **/

        public static void main(String[] args) throws IOException {
            /* Functons commented out for graphing LogLoss */ 
            //logLossSeries = new XYSeries("Log Loss");

            ArrayList<int[]> trainSet = new ArrayList<int[]>();

            //Reads trainSet file and sets to the ArrayList of int[]
            FileReader fr1 = new FileReader("./train-1.csv");
            trainSet = inputFile(fr1);

            //Creates LogRegression Learning model, with weights size one less due to the label column
            LogisticRegression logReg = new LogisticRegression(trainSet.get(1).length-1);
            
            //Train and tests trainset
            logReg.train(trainSet);
            System.out.println("Testing Trainset");
            logReg.test(trainSet);

            //Plot LogLoss of system
            //logReg.plotLogLoss(); 

            ArrayList<int[]> testSet = new ArrayList<int[]>();

            //Reads test file and sets to testSet ArrayList of int[]
            FileReader fr2 = new FileReader("./test-1.csv");
            testSet = inputFile(fr2);

            //Tests Trainset
            System.out.println("Testing Trainset");
            logReg.test(testSet);

        }


        
}


