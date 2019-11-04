<template>
    <div id="nav">
        {{test}}
        <div @click="goHome">Home</div> |
        <router-link to="/about">About</router-link>
        <div>
            <router-view></router-view>
        </div>
    </div>
</template>

<script lang="ts">
    import { Component, Vue, Watch } from 'vue-property-decorator';
    @Component
    export default class Index extends Vue {
        private test: string = '1234';
        @Watch('$route')
        private routeChange(val: any, old: any) {
            console.log(val);
            console.log(old);
        }

        private goHome() {
            this.$router.push({
                name: 'home',
            });
        }
    }
</script>

<style scoped>

</style>