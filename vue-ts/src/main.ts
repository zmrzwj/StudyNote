import Vue from 'vue';
import App from './App.vue';
import router from './router';
import store from './store';
import 'ant-design-vue/dist/antd.css';
import Ant from 'ant-design-vue';
const {
    LocaleProvider,
    Layout,
    Input,
    InputNumber,
    Pagination,
    Button,
    Switch,
    Radio,
    Checkbox,
    Select,
    Card,
    Form,
    Row,
    Col,
    Modal,
    Table,
    Tabs,
    Icon,
    Badge,
    Popover,
    Dropdown,
    List,
    Avatar,
    Breadcrumb,
    Steps,
    Spin,
    Menu,
    Drawer,
    Tooltip,
    Alert,
    Tag,
    Divider,
    DatePicker,
    TimePicker,
    Carousel,
    Upload,
    Progress,
    Skeleton,
    Popconfirm,
    message,
    notification,
} = Ant;

Vue.use(LocaleProvider);
Vue.use(Pagination);
Vue.use(Layout);
Vue.use(Input);
Vue.use(InputNumber);
Vue.use(Button);
Vue.use(Switch);
Vue.use(Radio);
Vue.use(Checkbox);
Vue.use(Select);
Vue.use(Card);
Vue.use(Form);
Vue.use(Row);
Vue.use(Col);
Vue.use(Modal);
Vue.use(Table);
Vue.use(Tabs);
Vue.use(Icon);
Vue.use(Badge);
Vue.use(Popover);
Vue.use(Dropdown);
Vue.use(List);
Vue.use(Avatar);
Vue.use(Breadcrumb);
Vue.use(Steps);
Vue.use(Spin);
Vue.use(Menu);
Vue.use(Drawer);
Vue.use(Tooltip);
Vue.use(Alert);
Vue.use(Tag);
Vue.use(Divider);
Vue.use(DatePicker);
Vue.use(TimePicker);
Vue.use(Upload);
Vue.use(Progress);
Vue.use(Skeleton);
Vue.use(Popconfirm);
Vue.use(Carousel);

Vue.prototype.$confirm = Modal.confirm;
Vue.prototype.$message = message;
Vue.prototype.$notification = notification;
Vue.prototype.$info = Modal.info;
Vue.prototype.$success = Modal.success;
Vue.prototype.$error = Modal.error;
Vue.prototype.$warning = Modal.warning;

Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount('#app');
