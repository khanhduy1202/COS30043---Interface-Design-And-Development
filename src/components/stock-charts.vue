<template>
    <div class="container">
        <h2 class="p-2 bg-dark text-white rounded-2">Stock Chart Parameters</h2><br>
        <form @submit.prevent="fetchChartData" class="needs-validation" novalidate>
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
            <button type="submit" class="btn btn-success">Fetch Chart Data</button>
        </form>
        <br />
        <div class="charts">
            <div v-if="chartLoading">Loading charts...</div>
            <div v-else>
                <div ref="candlestickChart" style="width: 100%; height: 400px;"></div>
                <div ref="boxplotChart" style="width: 100%; height: 400px;"></div>
            </div>
        </div>
    </div>
</template>

<script>
import axios from "axios";
import Plotly from "plotly.js-dist";

export default {
    data() {
        return {
            company: "",
            startDate: "",
            endDate: "",
            chartLoading: false,
        };
    },
    methods: {
        async fetchChartData() {
            this.chartLoading = true;
            const requestData = {
                company: this.company,
                startDate: this.startDate,
                endDate: this.endDate,
            };
            try {
                const response = await axios.post("http://localhost:8000/api/get_chart_data", requestData);
                const chartData = response.data;
                console.log(chartData)
                if (chartData && chartData.candlestick_data) {
                    await this.$nextTick();
                    setTimeout(() => {
                        this.renderCandlestickChart(chartData.candlestick_data);
                    }, 100);
                } else {
                    console.error("Invalid or missing candlestick data in the response.");
                }

                if (chartData && chartData.boxplot_data) {
                    this.renderBoxplotChart(chartData.boxplot_data);
                } else {
                    console.error("Invalid or missing boxplot data in the response.");
                }

                this.chartLoading = false;
            } catch (error) {
                console.error("Failed to fetch chart data:", error);
                this.chartLoading = false;
            }
        },
        renderCandlestickChart(candlestickData) {
            if (candlestickData.length === 0) {
                console.error("Candlestick data is empty.");
                return;
            }

            // Extract candlestick data from the first object in the array
            const candlestick = candlestickData[0];

            const candlestickChart = {
                x: candlestick.x,
                close: candlestick.close,
                high: candlestick.high,
                low: candlestick.low,
                open: candlestick.open,
                type: "candlestick",
            };

            const layout = {
                title: "Candlestick Chart",
                xaxis: { title: "Date" },
                yaxis: { title: "Price" },
            };

            Plotly.newPlot(this.$refs.candlestickChart, [candlestickChart], layout);
        },
        renderBoxplotChart(boxplotData) {
            const labels = Array.from({ length: boxplotData.length }, (_, i) => `Window ${i + 1}`);

            const boxplotTrace = {
                x: labels,
                y: boxplotData,
                type: "box",
            };

            const layout = {
                title: "Boxplot Chart",
                xaxis: { title: "Window" },
                yaxis: { title: "Closing Price" },
            };

            Plotly.newPlot(this.$refs.boxplotChart, [boxplotTrace], layout);
        },
    },
};
</script>
