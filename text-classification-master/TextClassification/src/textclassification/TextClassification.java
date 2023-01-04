/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package textclassification;

import static java.awt.PageAttributes.MediaType.A;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author Asus
 */
public class TextClassification {

    static String pathOfTrainLabels;
    static String PathOfTrainData;
    static String PathOfTestData;
    static String PathOfTestLabels;
    static String PathOfResults;

    public static void main(String[] args) throws IOException {
        Scanner reader = new Scanner(System.in);
        System.out.println("Enter the path of traindata.txt:");
        PathOfTrainData = reader.next();
        System.out.println("Enter the path of trainlabels.txt:");
        pathOfTrainLabels = reader.next();
        System.out.println("Enter the path of testdata.txt:");
        PathOfTestData = reader.next();
        System.out.println("Enter the path of testlabels.txt:");
        PathOfTestLabels = reader.next();
        System.out.println("Enter the path of results.txt:");
        PathOfResults = reader.next();
        PrintWriter writer = new PrintWriter(PathOfResults, "UTF-8");
        float accuracy = 0;
        float sumOfAll = 0;
        float Correct = 0;
        ArrayList<String> resultLabels1 = new ArrayList<>();
        ArrayList<String> resultLabels2 = new ArrayList<>();
        ArrayList<String> D = MakeDfromFile(PathOfTrainData);
        int[] C = {0, 1};
        ReturnTypeOfTrainMultinomialNB r = TrainMultinomialNB(C, D);
        ArrayList<String> Testdata1 = MakeDfromFile(PathOfTrainData);
        for (String line : Testdata1) {
            resultLabels1.add(ApplyMultiNomialNB(C, r.V, r.prior, r.condprob, line));
        }
        ArrayList<String> Testlabels1 = MakeDfromFile(pathOfTrainLabels);
        for (int i = 0; i < Testlabels1.size(); i++) {
            if (Testlabels1.get(i).equals(resultLabels1.get(i))) {
                Correct++;
            }
        }
        sumOfAll = Testlabels1.size();
        accuracy = Correct / sumOfAll;
        writer.println("traindata.txt and trainlabels.txt used for training and testing:");
        writer.println("accuracy:  " + accuracy);
        writer.println();
        System.out.println("traindata.txt and trainlabels.txt used for training and testing:");
        System.out.println("accuracy:  " + accuracy);
        accuracy = 0;
        sumOfAll = 0;
        Correct = 0;
        ArrayList<String> Testdata2 = MakeDfromFile(PathOfTestData);
        for (String line : Testdata2) {
            resultLabels2.add(ApplyMultiNomialNB(C, r.V, r.prior, r.condprob, line));
        }
        ArrayList<String> Testlabels2 = MakeDfromFile(PathOfTestLabels);
        for (int i = 0; i < Testlabels2.size(); i++) {
            if (Testlabels2.get(i).equals(resultLabels2.get(i))) {
                Correct++;
            }
        }
        sumOfAll = Testlabels2.size();
        accuracy = Correct / sumOfAll;
        writer.println("traindata.txt and trainlabels.txt used for training and testing on testdata.txt and testlabels.txt :");
        writer.println("accuracy:  " + accuracy);
        writer.close();
        System.out.println("traindata.txt and trainlabels.txt used for training and testing on testdata.txt and testlabels.txt :");
        System.out.println("accuracy:  " + accuracy);
    }

    public static ReturnTypeOfTrainMultinomialNB TrainMultinomialNB(int[] C, ArrayList<String> D) throws IOException {
        ArrayList<String> V = ExtractVocabulary(D);
        double[][] T = new double[2][V.size()];
        double condprob[][] = new double[V.size()][2];
        double N = CountDocs(D);
        double Nc[] = new double[2];
        double prior[] = new double[2];
        ArrayList<String> text0 = null;
        ArrayList<String> text1 = null;
        for (int c : C) {
            Nc[c] = CountDocsInClass(String.valueOf(c));
            prior[c] = Nc[c] / N;
            if (c == 0) {
                text0 = ConcatenateTextOfAllDocsInClass(D, "0");
            }
            if (c == 1) {
                text1 = ConcatenateTextOfAllDocsInClass(D, "1");
            }
            for (int i = 0; i < V.size(); i++) {
                String t = V.get(i);
                if (c == 0) {
                    T[0][i] = CountTokensOfTerm(text0, t);
                }
                if (c == 1) {
                    T[1][i] = CountTokensOfTerm(text1, t);
                }
            }
            double Denominator = V.size();
            for (int i = 0; i < V.size(); i++) {
                Denominator = Denominator + T[c][i];
            }
            for (int i = 0; i < V.size(); i++) {
                condprob[i][c] = (T[c][i] + 1) / Denominator;
            }
        }
        return new ReturnTypeOfTrainMultinomialNB(V, prior, condprob);
    }

    public static String ApplyMultiNomialNB(int[] C, ArrayList<String> V, double[] prior, double[][] condprob, String d) {
        ArrayList<String> W = ExtractTokensFromDoc(V, d);
        double[] score = new double[2];
        for (int c : C) {
            score[c] = Math.log10(prior[c]);
            for (String t : W) {
                for (int i = 0; i < V.size(); i++) {
                    if (V.get(i).equals(t)) {
                        score[c] += Math.log10(condprob[i][c]);
                        break;
                    }
                }
            }
        }
        if (score[0] > score[1]) {
            return "0";
        } else {
            return "1";
        }
    }

    public static ArrayList<String> ExtractVocabulary(ArrayList<String> D) {
        ArrayList<String> V = new ArrayList<>();
        for (String line : D) {
            String[] words = line.split(" ");
            for (String word : words) {
                if (!V.contains(word)) {
                    V.add(word);
                }
            }
        }
        return V;
    }

    public static int CountDocs(ArrayList<String> D) {
        return D.size();
    }

    public static int CountDocsInClass(String c) throws FileNotFoundException, IOException {
        int NC = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(pathOfTrainLabels))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.equals(c)) {
                    NC++;
                }
            }
        }
        return NC;
    }

    public static int CountTokensOfTerm(ArrayList<String> textc, String t) {
        int Tct = 0;
        for (String line : textc) {
            String[] words = line.split(" ");
            for (String word : words) {
                if (t.equals(word)) {
                    Tct++;
                }
            }
        }
        return Tct;
    }

    public static ArrayList<String> ConcatenateTextOfAllDocsInClass(ArrayList<String> D, String c) throws FileNotFoundException, IOException {
        ArrayList<String> textc = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(pathOfTrainLabels))) {
            String line;
            int linenumber = 0;
            while ((line = br.readLine()) != null) {
                if (line.equals(c)) {
                    textc.add(D.get(linenumber));
                }
                linenumber++;
            }
        }
        return textc;
    }

    public static ArrayList<String> ExtractTokensFromDoc(ArrayList<String> V, String d) {
        ArrayList<String> W = new ArrayList<>();
        String[] words = d.split(" ");
        for (String word : words) {
            if (V.contains(word) && !(W.contains(word))) {
                W.add(word);
            }
        }
        return W;
    }

    public static ArrayList<String> MakeDfromFile(String path) throws FileNotFoundException, IOException {
        ArrayList<String> D = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                D.add(line);
            }
        }
        return D;
    }
}
