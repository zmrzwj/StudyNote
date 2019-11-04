import Vue from 'vue';
import Router from 'vue-router';
import Index from './views/index.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'index',
      component: Index,
        children: [
            {
                path: 'about',
                name: 'about',
                component: () => import(/* webpackChunkName: "about" */ './views/About.vue'),
            },
            {
                path: 'home',
                name: 'home',
                component: () => import(/* webpackChunkName: "about" */ './views/Home.vue'),
            },
        ],
    },
  ],
});
