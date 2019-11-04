<template>
  <div class="home">
    <div>{{side}}</div>
      <a-input placeholder="请输入名字"></a-input>
    <HelloWorld msg="Welcome to Your Vue.js + TypeScript App"/>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import HelloWorld from '@/components/HelloWorld.vue'; // @ is an alias to /src

@Component({
  components: {
    HelloWorld,
  },
})
export default class Home extends Vue {
    private side: string = '我是hello world';
}
</script>
