<template>
    <div class="container">
        <h2 class="p-2 bg-dark text-white rounded-2">Look up stock data</h2><br>
        <form @submit.prevent="fetchStockData" class="needs-validation" novalidate>
            <div class="form-group">
                <label for="company">Company:</label>
                <input type="text" id="company" v-model="company" class="form-control" required>
                <div class="invalid-feedback">Company is required.</div>
            </div>
            <br />
            <div class="form-group">
                <label for="startDate">Start Date:</label>
                <input type="date" id="startDate" v-model="startDate" class="form-control" required>
                <div class="invalid-feedback">Start Date is required.</div>
            </div>
            <br />
            <div class="form-group">
                <label for="endDate">End Date:</label>
                <input type="date" id="endDate" v-model="endDate" class="form-control" required>
                <div class="invalid-feedback">End Date is required.</div>
            </div>
            <br />
            <button type="submit" class="btn btn-success">Fetch Data</button>
        </form>
        <br />
        <stock-data :stockData="stockData"></stock-data>
    </div>
</template>
  
  

<script>
import axios from "axios";
import StockData from '@/components/stock-data.vue';

export default {
    data() {
        return {
            // Define data properties for user input (e.g., company, dates, etc.)
            company: "",
            startDate: "",
            endDate: "",
            stockData: []
        };
    },
    methods: {
        async fetchStockData() {
            // Create an object to store the required parameters
            const requestData = {
                company: this.company,
                startDate: this.startDate,
                endDate: this.endDate,
            };
            try {
                // Send a POST request to retrieve data from backend server
                // eslint-disable-next-line no-unused-vars
                const response = await axios.post("http://localhost:8000/api/fetch_stock_data", requestData);
                console.log("Received data:", response.data);
                this.stockData = response.data;
                this.$emit("data-updated", this.stockData);
            } catch (error) {
                console.error("Failed to fetch data!", error);
            }
        },
    },
    components: {
        StockData,
    }
};
</script>