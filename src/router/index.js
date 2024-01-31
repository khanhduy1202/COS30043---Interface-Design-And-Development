import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '@/components/get-data.vue'
import Predict from '@/components/stock-predict.vue'
import Charts from '@/components/stock-charts.vue'

Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'getdata',
    component: Home
  },
  {
    path: '/Predict',
    name: 'predictstock',
    component: Predict
  },
  {
    path: '/Charts',
    name: 'stockcharts',
    component: Charts
  }
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
