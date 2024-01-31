<template>
	<div>
		<div class="tb_overflow">
			<table class="table table-striped table-bordered">
				<thead>
					<tr>
						<th>Date</th>
						<th>Open</th>
						<th>High</th>
						<th>Low</th>
						<th>Close</th>
						<th>Adj Close</th>
						<th>Volume</th>
					</tr>
				</thead>
				<tbody>
					<tr v-for="row in displayedData" :key="row.Date">
						<td>{{ row.Date }}</td>
						<td>{{ row.Open }}</td>
						<td>{{ row.High }}</td>
						<td>{{ row.Low }}</td>
						<td>{{ row.Close }}</td>
						<td>{{ row['Adj Close'] }}</td>
						<td>{{ row.Volume }}</td>
					</tr>
				</tbody>
			</table>
		</div>
		<div class="text-center">
			<button @click="previousPage" :disabled="currentPage === 0" class="btn btn-primary">Previous Page</button>
			<span class="mx-2">{{ currentPage + 1 }}</span>
			<button @click="nextPage" :disabled="currentPage === totalPages - 1" class="btn btn-primary">Next Page</button>
		</div>
	</div>
</template>
  
<script>
export default {
	name: 'StockData',
	props: {
		stockData: Array, // Prop to receive the stock data from get data
	},
	data() {
		return {
			itemsPerPage: 30,
			currentPage: 0,
		};
	},
	computed: {
		totalPages() {
			return Math.ceil(this.stockData.length / this.itemsPerPage);
		},
		displayedData() {
			const startIndex = this.currentPage * this.itemsPerPage;
			const endIndex = startIndex + this.itemsPerPage;
			return this.stockData.slice(startIndex, endIndex);
		},
	},
	methods: {
		previousPage() {
			if (this.currentPage > 0) {
				this.currentPage--;
			}
		},
		nextPage() {
			if (this.currentPage < this.totalPages - 1) {
				this.currentPage++;
			}
		},
	},
};
</script>
<style>
.tb_overflow {
	overflow-x: scroll;
	overflow-y: scroll;
}
</style>