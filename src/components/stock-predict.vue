<template>
    <div class="container">
        <form @submit.prevent="predictStockData" class="needs-validation row" novalidate>
            <div class="form-group col-md-6 col-sm-12">
                <h2 class="p-2 bg-dark text-white rounded-2">Training Parameters</h2>
                <div class="form-group">
                    <label for="company">Company:</label>
                    <input type="text" id="company" v-model="company" class="form-control" required>
                    <div class="invalid-feedback">Company is required.</div>
                </div><br>
                <div class="form-group">
                    <label for="startDate">Training Start Date:</label>
                    <input type="date" id="startDate" v-model="startDate" class="form-control" required>
                    <div class="invalid-feedback">Training Start Date is required.</div>
                </div><br>
                <div class="form-group">
                    <label for="priceValue">Price Value:</label>
                    <select id="priceValue" v-model="priceValue" class="form-control" required>
                        <option value="Open">Open</option>
                        <option value="High">High</option>
                        <option value="Low">Low</option>
                        <option value="Close">Close</option>
                        <option value="Adj Close">Adj Close</option>
                        <option value="Volume">Volume</option>
                    </select>
                    <div class="invalid-feedback">Price Value is required.</div>
                </div><br>
                <div class="form-group">
                    <label for="predictionDays">Training Days Size:</label>
                    <input type="number" id="predictionDays" v-model="predictionDays" class="form-control" required>
                    <div class="invalid-feedback">Prediction Days is required.</div>
                </div><br>
            </div>
            <div class="form-group col-md-6 col-sm-12">
                <h2 class="p-2 bg-dark text-white rounded-2">Model Parameters</h2>
                <div class="form-group">
                    <label for="epochs">Training Cycles (how many times you want to train the model):</label>
                    <input type="number" id="epochs" v-model="epochs" class="form-control" required>
                    <div class="invalid-feedback"></div>
                </div><br>
                <div class="form-group">
                    <label for="batchSize">Batch Size (how many units of data in a training cycle):</label>
                    <input type="number" id="batchSize" v-model="batchSize" class="form-control" required>
                    <div class="invalid-feedback"></div>
                </div><br>
                <div class="form-group">
                    <label for="numOfDays">Number of Days to predict:</label>
                    <input type="number" id="numOfDays" v-model="numOfDays" class="form-control" required>
                    <div class="invalid-feedback">Number of Days is required.</div>
                </div><br>
            </div><br>
            <div class="text-center">
                <button type="submit" class="btn btn-success col-2">Predict</button>
            </div>
        </form><br>
        <div class="predictions">
            <h3>Predictions:</h3>
            <ul v-if="dataLoading">
                <li>Loading...</li>
            </ul>
            <table class="table table-striped table-bordered">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Predicted Price</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="(prediction, index) in predictedStock" :key="index">
                        <td>{{ prediction.date }}</td>
                        <td>{{ prediction.price }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>
  
  
<script>
import axios from "axios";

export default {
    data() {
        return {
            // dataframe properties
            company: "",
            startDate: "",
            priceValue: "",
            predictionDays: 30,
            //model properties
            units: 50,
            nLayers: 2,
            dropout: 0.2,
            epochs: 5,
            batchSize: 32,
            numOfDays: 1,

            predictedStock: [],
            dataLoading: false
        };
    },
    methods: {
        async predictStockData() {
            this.dataLoading = true;
            // Create an object to store the required parameters
            const requestData = {
                company: this.company,
                startDate: this.startDate,
                priceValue: this.priceValue,
                predictionDays: this.predictionDays,
                // Include model parameters
                units: this.units,
                nLayers: this.nLayers,
                dropout: this.dropout,
                epochs: this.epochs,
                batchSize: this.batchSize,
                numOfDays: this.numOfDays,
            };
            try {
                // Send a POST request to retrieve data from backend server
                // eslint-disable-next-line no-unused-vars
                const response = await axios.post("http://localhost:8000/api/predict_stock_data", requestData);
                this.dataLoading = false;  // Data is ready
                this.predictedStock = response.data
                console.log("Predicted Stock Data:", this.predictedStock);
                this.$emit("predicted-stock-updated", this.predictedStock);

            } catch (error) {
                console.error("Failed to fetch data!", error);
            } finally {
                this.dataLoading = false;
            }
        },
    },
};
</script>