<template>
  <div id="app" class="container">
    <HomeView />
    <nav>
      <router-link to="/">Home</router-link> |
      <router-link to="/Predict">Predict</router-link> |
      <router-link to="/Charts">Charts</router-link>
    </nav>
    <router-view />
  </div>
</template>

<script>
import HomeView from '@/views/HomeView.vue'

export default {
  components: {
    HomeView
  },
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

nav {
  padding: 30px;
}

nav a {
  font-weight: bold;
  color: #2c3e50;
}

nav a.router-link-exact-active {
  color: #42b983;
}
</style>
