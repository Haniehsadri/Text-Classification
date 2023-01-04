/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package textclassification;

import java.util.ArrayList;

/**
 *
 * @author Asus
 */
public class ReturnTypeOfTrainMultinomialNB {
    ArrayList<String> V;
    double[] prior;
    double[][] condprob;
    public ReturnTypeOfTrainMultinomialNB(ArrayList<String> V, double [] prior, double [][] condprob){
        this.V=V;
        this.prior=prior;
        this.condprob=condprob;
    }
}
