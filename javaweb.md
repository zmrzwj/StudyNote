## 1、tomcat

### 1.1.tomcat下载：

​	https://tomcat.apache.org/download-80.cgi
​	下载zip文件

### 1.2.配置

​	conf/tomcat-users.xml
​	添加：
​	<user username="admin" password="admin" roles="manager-gui,admin-gui"/>
​	conf/web.xml
​	添加：
​	<init-param>
​            <param-name>compilerSource</param-name>
​            <param-value>1.8</param-value>
​        </init-param>
​        <init-param>
​            <param-name>compilerTargetVM</param-name>
​            <param-value>1.8</param-value>
​        </init-param>
​	conf/server.xml可以修改接口

### 1.3.运行

​	./startup.sh
​	如果出现startup.sh: command not found
​        需要更改startup.sh权限
​	sudo chmod 755 *.sh //需要在tomcat-xxx/bin/目录下执行
​	执行成功访问http://localhost:8080/
​	./shutdown.sh关闭tomcat

## 2、创建Maven

​	groupId一般分为多个段，这里我只说两段，第一段为域，第二段为公司名称。域又分为org、com、cn等等许多。如：top.hellozwj
​	artifactId设置为项目的名称;
​	使用阿里maven：
​	修改pom.xml:
​		<repositories><!-- 代码库 -->
​        		<repository>
​          			<id>maven-ali</id>
​            			<url>http://maven.aliyun.com/nexus/content/groups/public//</url>
​            			<releases>
​                			enabled>true</enabled>
​            			</releases>
​            			<snapshots>
​                			<enabled>true</enabled>
​                			<updatePolicy>always</updatePolicy>
​                			<checksumPolicy>fail</checksumPolicy>
​            			</snapshots>
​        		</repository>
​    		</repositories>
​	如果想修改settings文件的话：settings.xml的默认路径就：个人目录/.m2/settings.xml
​		Mac下在～/.m2/下,如果没有就自己创建
​			内容：<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
​      				xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
​      				xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
​                          	https://maven.apache.org/xsd/settings-1.0.0.xsd">
  				<localRepository/>
  				<interactiveMode/>
  				<usePluginRegistry/>
  				<offline/>
  				<pluginGroups/>
  				<servers/>
  				<mirrors/>
  				<proxies/>
  				<profiles/>
  				<activeProfiles/>
​			</settings>
​			

## 3、下载离线maven

http://maven.apache.org/download.cgi
Mac 下载-bin.tar.gz解压到zwj/applictions
在使用idea创建maven时，更改默认的maven,选择解压的目录就到apache-maven-3.5.4
同时settings也要改为apache-maven-3.5.4/conf/settings.xml
<mirrors>中添加
	<mirror>
        	<id>nexus-aliyun</id>
        	<mirrorOf>*</mirrorOf>
        	<name>Nexus aliyun</name>
        	<url>http://maven.aliyun.com/nexus/content/groups/public</url>
    	</mirror>

## 4、IDEA设置tomcat

​	run->Edit Configurations
​	点击：+,点击：tomcat server,点击：local
​	然后配置tomcat，选择解压的tomcat目录
​	如果出现错误：no artifacts configured：
​	打开file->Project Structure,选择Aritfacts选项， 选择Web Application:Exploded,修改命名name,在右边avilable elements下选中要添加的项目，点击+，选择directory content,选择项目的目录.再次进入Edit Configration->Deployment，点击+号，即可看到Artifact选项了.(注意：Aritfact一定要是：xxx:war exploded)
​	war模式：将WEB工程以包的形式上传到服务器 ；
​	war exploded模式：直接把文件夹、jsp页面 、classes等等移到Tomcat 部署文件夹里面，进行加载部署，一般在开发的时候也是用这种方式。
​	就可以运行项目了，但什么都没有浏览器会显示404

## 5.创建webapp

选择Maven项目时，勾选create from archetype,并选中org.apache.maven.archetypes:maven-archetype-webapp.
然后创建完项目后会加载很多依赖，会很慢，加载完后才会出现所有目录结构，不然只有pom.xml。
最后下载离线的Maven，在创建项目时更改为自己下载的maven解压后的目录，settings.xml也要更改为自己的settings.xml在maven目录的conf目录里。

## 6.创建的项目main下没有java目录

解决：   选择File->Project Structure
	选择Modules选项卡下面的Sources项，在main文件夹上右键，选择New Folder...并点击OK
	输入要创建的文件夹名称java，并点击OK继续
	在创建好的java文件夹上右键选择Sources项将该文件夹标记为源文件夹
	我们发现java文件夹已经由黄色变成了蓝色，我们点击OK按钮表示设置完成（蓝色就对了）
解决2:在main文件夹上右键,直接创建目录，即可。

## 7.Servlet should have a mapping

请添加<servlet-mapping>
    	<servlet-name>HelloWorld</servlet-name>
    	<url-pattern>/hello</url-pattern>
     </servlet-mapping>

## 8.创建断点

<servlet>下添加
<load-on-startup>1</load-on-startup>

## 9.@WebServlet

@WebServlet(
        name = "hello",
        urlPatterns = {"/hello","/greeting"},
        loadOnStartup = 1
)可以代替web.xml的servlet的配置

## 10.javax.inject.jar：依赖注入非常方便的jar包.

用过Spring框架的我们都知道，每当生成依赖注入的时候，我们都必须生成相应类的set方法，而且要在set方法上面写上@Autowired，才能实现依赖注入.

## 11.scope=compile

scope=compile的情况（默认scope),也就是说这个项目在编译，测试，运行阶段都需要这个artifact对应的jar包在classpath中。
scope=provided的情况，则可以认为这个provided是目标容器已经provide这个artifact。换句话说，它只影响到编译，测试阶段。在编译测试阶段，我们需要这个artifact对应的jar包在classpath中，而在运行阶段，假定目标的容器（比如我们这里的liferay容器）已经提供了这个jar包，所以无需我们这个artifact对应的jar包了。
即scope=provided，打包不会打包进去。



## 12.Artifact 是maven中的一个概念，表示某个module要如何打包。



## 13、scope的分类

compile
默认就是compile，什么都不配置也就是意味着compile。compile表示被依赖项目需要参与当前项目的编译，当然后续的测试，运行周期也参与其中，是一个比较强的依赖。打包的时候通常需要包含进去。
test
scope为test表示依赖项目仅仅参与测试相关的工作，包括测试代码的编译，执行。比较典型的如junit。
runntime
runntime表示被依赖项目无需参与项目的编译，不过后期的测试和运行周期需要其参与。与compile相比，跳过编译而已，说实话在终端的项目（非开源，企业内部系统）中，和compile区别不是很大。
	    比较常见的如JSR×××的实现，对应的API jar是compile的，具体实现是runtime的，compile只需要知道接口就足够了。
	    oracle jdbc驱动架包就是一个很好的例子，一般scope为runntime。另外runntime的依赖通常和optional搭配使用，optional为true。我可以用A实现，也可以用B实现。
provided
provided意味着打包的时候可以不用包进去，别的设施(Web Container)会提供。事实上该依赖理论上可以参与编译，测试，运行等周期。相当于compile，但是在打包阶段做了exclude的动作。
system
从参与度来说，也provided相同，不过被依赖项不会从maven仓库抓，而是从本地文件系统拿，一定需要配合systemPath属性使用。

## 14.RSS 简单信息聚合



## 15.SOAP(simple object access protocol)简单对象访问协议



## 16.ORM对象关系映射

面向对象编程语言



## 17.应用服务器内置连接池

应用服务器有专门用于管理连接池的内建系统，可以改善应用程序中数据库连接的性能。对应管理这些连接的应用服务器，必须在应用服务器类加载器中而不是web应用类加载器中加载JDBC驱动。
在mysql官网下载mysql JDBC,拿到JAR文件,把它放到tomcat\lib下。修改tomcat\conf\context.xml
在<Context>里添加
<Resource
            name="jdbc/GZXS"
            type="javax.sql.DataSource"
            maxActive="100"
            maxIdle="30"
            maxWait="5000"
            username="root"
            password="123456"
            driverClassName="com.mysql.jdbc.Driver"
            url="jdbc:mysql://localhost:3306/GZXS?autoReconnect=true&amp;useSSl=false"
    />
//maxIdle连接池中最少空闲30个连接，maxWait连接池连接用完时新请求等待时间，name为Resource的名字,&amp;是&的转义符直接使用&会报错。
mysql-connector-java-bin.jar与mysql-connector-java.jar
使用上是没区别的一样的,都可以用,带-bin的文件里在编译的的时候里面多了几个编译用的校验文件而已.
数据源在tomcat /conf/content/xml与/conf/server.xml的区别
server.xml不可动态重新加载资源，要修改这个文件，就要重启才能重新加载。context.xml文件，tomcat一旦文件修改(时间戳改变)，就会重新加载这个文件，不需要重启服务器。



## 18.ubuntu 安装tomcat9

下载：apache-tomcat-9.0.11.tar.gz
解压到/usr/local
修改tomcat/bin下的startup.sh文件，在最后添加
export JAVA_HOME=/usr/local/java
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH

启动tomcat
  ./bin/startup.sh
  curl localhost:8080
关闭tomcat
  ./bin/shuntdown.sh



## 19.SpringMVC 中 url-patter 与 @RequestMapping 的对应问题

利用 SpringMVC 响应前端发起的请求时，其完整 url 会按照 DispatcherServlet 指定的 url 格式进行匹配、修剪，去掉<url-pattern>指定的上下文部分，剩余部分 url'，再由注解 @RequestMapping 转到 Controller 特定的方法上，执行具体的处理。
前端请求的完整 url：
	http://localhost:8088/aaa/bbb/ccc?myParam=myValue
servlet 指定的 url 格式：
	<url-pattern>/aaa/*</url-pattern> <!-- *一定要加不然访问不了 -->
则 @RequestMapping 的正确配置为：
	@RequestMapping(value="/bbb/ccc")
	public String demoMethodSignature(String myParam) 
	

## 20.Spring项目

Controller层：负责具体业务模块流程的控制，即调用Service层的接口来控制业务流程。负责url映射（action）。
Dao层：负责数据持久化，与数据库进行联络的任务都封装在其中，Dao层的数据源以及相关的数据库连接参数都在Spring配置文件中进行配置。
Entity层：java对象，与数据库表一一对应，是其对应的实现类。即一个Entity就是对应表中的一条记录。
Service层：建立在DAO层之上，Controller层之下。调用Dao层的接口，为Controller层提供接口。负责业务模块的逻辑应用设计，首先设计接口，再设计其实现的类。
View层：表示层，负责前端jsp页面表示。



## 21.JDBC连接数据库java.sql.Connection

<dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
</dependency>
import java.sql.*;
//声明Connection对象
         Connection con;
         //驱动程序名
        String driver = "com.mysql.jdbc.Driver";
         //URL指向要访问的数据库名mydata
         String url = "jdbc:mysql://localhost:3306/sqltestdb";
         //MySQL配置时的用户名
         String user = "root";
        //MySQL配置时的密码
         String password = "123456";
             //加载驱动程序
             Class.forName(driver);
             //1.getConnection()方法，连接MySQL数据库！！
             con = DriverManager.getConnection(url,user,password);
	     Statement stat = conn.createStatement()创建一个Statement对象来将SQL语句发送到数据库
	    //要执行的SQL语句
             String sql = "select count(1) from VIEW_JLZDH_OB_DL_DAY";
	    //ResultSet类，用来存放获取的结果集！！
            ResultSet rs = statement.executeQuery(sql);
	    while (rs.next()){
                //输出结果
                System.out.println(rs.getString("count(1)"));
            }



## 22.mac IDEA try catch自动添加，option+command+t



## 23.preparedStatement与Statement

preparedStatement是预编译的，对批处理可以大大提高效率，Statement只执行一次性存取。preparedStatement用于执行带参数的预编译SQL, Statement执行不带参数的简单SQL。执行的对象较多时，preparedStatement会降低运行时间。企业更喜欢preparedStatement，因为它更安全，传递给preparedStatement的对象参数可以被强制进行类型转换。Statement每次执行一个SQL，都会对它进行解析和编译。preparedStatement和Statement都需要调用close函数关闭，不应等待对象自动关闭。



## 24.java:comp/env/

这是J2EE环境的定义，代表了当前J2EE应用的环境，使用这样的方式，必须设置当前应用环境到资源名的映射。



## 25.JDBC-DataSource(数据库连接池)

JDBC的数据库连接池使用javax.sql.DataSource来表示，它只是一个接口，该接口由应用服务器实现，或开源组织（DBCP,C3P0）.
DBCP:
需要两个依赖：commons-dbcp.jar，commons-pool.jar
Tomcat的连接池正是采用这个连接池实现的，可以和应用服务器整合使用，也可以由程序单独使用。
C3P0:
C3p0性能更胜一筹，hibernate推荐使用该连接池，c3p0连接池不仅可以自动清理不使用的connection，还可以自动清理statement和resultset.



## 26.idea创建springboot项目

Maven的Snapshot版本与Release版本

1. Snapshot版本代表不稳定、尚处于开发中的版本
2. Release版本则代表稳定的版本
1）选择File –> New –> Project –>Spring Initialer –> 点击Next 
2）自己修改 Group 和 Artifact 字段名 –>点击next 
3）选择web，勾选web, spring boot可以选择spring boot版本
4）点击finish 
5)等待编译完成(这个貌似翻墙会好点,要不然下载依赖特别慢。。。)



## 27.application.properties

原来spring boot2.0之后，server.context-path上下文的配置改为了server.servlet.context-path



## 28.idea spring打包为war包

<packaging>jar</packaging>改为
<packaging>war</packaging>
主程序文件修改如下：
@SpringBootApplication
public class XsApplication extends SpringBootServletInitializer {

    @Override
    protected SpringApplicationBuilder configure(
            SpringApplicationBuilder application) {
        return application.sources(XsApplication.class);
    }


    public static void main(String[] args) {
        SpringApplication.run(XsApplication.class, args);
    }
}
修改pom.xml
<dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <!-- 移除嵌入式tomcat插件 -->
            <exclusions>
                <exclusion>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-tomcat</artifactId>
                </exclusion>
            </exclusions>
</dependency>
或
<dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-tomcat</artifactId>
            <scope>provided</scope>
</dependency>
在project settings点击artifacts:
点击+，选择web application archive,勾选include in project build(或Build on make,视idea版本而定，勾选了才会产生war包不然就是个目录)，在下面点击+号选择Directory Content，选择你的项目目录整个目录哟，点击ok.
点击build->Build Artifacts，选xs.war(war与war exploded的区别见6，如果没有先build一次idea会自动生成war和war exploded)，如果是第二次build就选择rebuild。

server.contextPath=/prefix/其中prefix为前缀名。这个前缀会在war包中失效，取而代之的是war包名称。

问题：Error:(12, 8) java: 无法访问javax.servlet.ServletException 找不到javax.servlet.ServletException的类文件
Pom.xml添加
<dependency>
    <groupId>javax</groupId>
    <artifactId>javaee-api</artifactId>
    <version>7.0</version>
</dependency>



## 29.如果使用spring-boot的服务器

在Edit Configurations时选择Spring-boot,Main class填写
com.cddw.com.xs.XsApplication

## 30、fastjson将容器转为字符串

<dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>fastjson</artifactId>
        <version>1.2.50</version>
</dependency>



## 31、创建普通maven工程

idea:
点击Create New Project ->然后选择Maven->新建普通的Java工程，因此选择-quickstart,如果是新建Java web工程就选择-webapp
->填写GroupID与ArtifactId (GroupID :com.test一般填域名, ArtifactId :一般为项目名)
-> 点击下一步 -> 填写工程名和保存路径 -> 点击下一步，工程建好后，会自动下载Maven相关的包
创建运行环境：
点击Run->Edit Configurations ，接着点击 + ，在新出来的界面，点击Application



## 32.spring boot jdbc连接池

<dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-jdbc</artifactId>
</dependency>
如果添加以上依赖那么已默认启用了数据库链接池
常用连接池：dbcp,dbcp2，druid,c3p0,hikari
hikari是性能最强的连接池，优化力度大。
#指定数据源
spring.datasource.type=com.zaxxer.hikari.HikariDataSource
#指定连接维护的最小空闲连接数，当使用HikariCP时指定.
spring.datasource.hikari.minimum-idle=10
#指定连接池最大的连接数，包括使用中的和空闲的连接.
spring.datasource.hikari.maximum-pool-size=30
#指定连接池等待连接返回的最大等待时间，毫秒单位.
spring.datasource.hikari.max-lifetime=500000



## 33.springboot项目结构

在com.xxx.xxx下
entity
    usr类
service
    使用dao的数据做逻辑操作
dao
    usrDao为接口
    usrDaoImpl实现其接口
controller
    实现接口映射@REstController



## 34.springboot拦截器

与SpringBootApplication同级目录
创建：TokenInterceptor类
@Component
public class TokenInterceptor implements HandlerInterceptor {

	@Override
	public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler){}
}
再创建TokenConfig：
@Configuration
public class TokenConfig implements WebMvcConfigurer {
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new TokenInterceptor()).addPathPatterns("/**").excludePathPatterns("/login");
    }
}
注意：这个路径不考虑server.servlet.context-path=/xxx添加的前缀。

@ControllerAdvice
统一异常处理
需要配合@ExceptionHandler使用，当将异常抛到controller时,可以对异常进行统一处理



## 35.纯Java项目使用HikariCP

Java 8
<dependency>
    <groupId>com.zaxxer</groupId>
    <artifactId>HikariCP</artifactId>
    <version>2.7.9</version>
</dependency>

private String url = "jdbc:mysql://127.0.0.1:3306/magnetic?characterEncoding=utf8&useSSL=false";
    private String username = "root";
    private String password = "123456";

    private HikariDataSource ds;
    
    public void start(){
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl(this.url);
        config.setUsername(this.username);
        config.setPassword(this.password);
        config.setDriverClassName("com.mysql.jdbc.Driver");
        //config.setDataSourceClassName();
    
        //是否自定义配置，为true时下面两个参数才生效
        config.addDataSourceProperty("cachePrepStmts","true");
        //连接池大小默认25，官方推荐250-500
        config.addDataSourceProperty("prepStmtCacheSize","250");
        //单条语句最大长度默认256，官方推荐2048
        config.addDataSourceProperty("prepStmtCacheSqlLimit", "2048");
        //新版本MySQL支持服务器端准备，开启能够得到显著性能提升
        config.addDataSourceProperty("useServerPrepStmts", "true");
        //池中维护的最小空闲连接数,minimumIdle强烈建议不要配置、默认值与maximumPoolSize相同
        //config.addDataSourceProperty("minimumIdle", "10");
        //池中最大连接数，包括闲置和使用中的连接
        config.addDataSourceProperty("maximumPoolSize", "20");
        //HikariCP将尝试通过仅基于jdbcUrl的DriverManager解析驱动程序，但对于一些较旧的驱动程序，还必须指定driverClassName
        config.addDataSourceProperty("driverClassName", "com.mysql.jdbc.Driver");
    
        this.ds = new HikariDataSource(config);
    }


报错：java.sql.SQLTransientConnectionException: HikariPool-1 - Connection is not available, request timed out after 30000ms.
原因:连接泄漏(在从池中借用之后连接没有关闭)。
hikariDataSource.setIdleTimeout(60000);           //连接允许在池中闲置的最长时间
hikariDataSource.setConnectionTimeout(60000);     //等待来自池的连接的最大毫秒数
hikariDataSource.setValidationTimeout(3000);      //连接将被测试活动的最大时间量
hikariDataSource.setLoginTimeout(5);              //
hikariDataSource.setMaxLifetime(60000);           //池中连接最长生命周期



## 36.spring缓存方案

有guava cache、redis、tair、memcached等。
Spring Boot默认情况下使用ConcurrentMapCacheManager作为缓存技术。
@Cacheable注解可以使用缓存



## 37.spring boot定时任务

//表示每个星期1中午5点 "0 0 5 ? * 1"
@Service
public class ScheduledService{
    @Scheduled(cron = "0 0 5 ? * 1")
    public void cronService(){}
}
开启定时任务注解@EnableScheduling
@SpringBootApplication
@EnableScheduling
public class EhomeApplication extends SpringBootServletInitializer{}



## 38.依赖注入有三种方式：

变量（filed）注入
构造器注入
set方法注入

变量（filed）注入
    @Autowired
    UserDao userDao;

构造器注入
    final
    UserDao userDao;
    @Autowired
    public UserServiceImpl(UserDao userDao) {
        this.userDao = userDao;
    }

set方法注入
    private UserDao userDao;
    @Autowired
    public void setUserDao (UserDao userDao) {
        this.userDao = userDao;
    }

如果这个类使用了依赖注入的类，那么这个类摆脱了这几个依赖必须也能正常运行。然而使用变量注入的方式是不能保证这点的。 
所以变量方式注入应该尽量避免，使用set方式注入或者构造器注入，这两种方式的选择就要看这个类是强制依赖的话就用构造器方式，选择依赖的话就用set方法注入。



## 39.Spring Boot动态定时任务

@Autowired
JdbcTemplate jdbcTemplate;
private ScheduledFuture<?> future; 
future = threadPoolTaskScheduler.schedule(new MyRunnable(), new CronTrigger("0/5 * * * * *"));
private class MyRunnable implements Runnable {
       @Override
       public void run() {
	   String sql = "update CLOCK set isOn = false where clock_id= ?;";
           jdbcTemplate.update(sql,this.clock_id);
           System.out.println("DynamicTask.MyRunnable.run()，" + new Date());
       }
}



## 40.maven工程连接c3p0

<!-- https://mvnrepository.com/artifact/com.mchange/c3p0 -->
        <dependency>
            <groupId>com.mchange</groupId>
            <artifactId>c3p0</artifactId>
            <version>0.9.5.2</version>
        </dependency>

    public static ComboPooledDataSource dataSource = new ComboPooledDataSource();
    static {
        try {
            dataSource.setDriverClass("com.mysql.jdbc.Driver");
            dataSource.setJdbcUrl("jdbc:mysql://39.98.182.112:3306/EHome?characterEncoding=utf8&useSSL=false");
            dataSource.setUser("root");
            dataSource.setPassword("uestc@123456");
    
            //连接池每隔60秒自动检测数据库连接情况，如果断开则自动重连。
            dataSource.setTestConnectionOnCheckin(true);
            dataSource.setIdleConnectionTestPeriod(60);
        } catch (PropertyVetoException e){
            e.printStackTrace();
        }
    }
    public static Connection getConnection(){
        try {
            return dataSource.getConnection();
        }catch (SQLException e){
            e.printStackTrace();
            throw new RuntimeException();
        }
    }



## 41.spring基础知识

1）POJO
POJO（Plain Ordinary Java Object）简单的Java对象，实际就是普通JavaBeans，是为了避免和EJB混淆所创造的简称。其中有一些属性及其getter setter方法的类,没有业务逻辑。
2）IOC
Ioc—Inversion of Control，即“控制反转”，不是什么技术，而是一种设计思想。Ioc意味着将你设计好的对象交给容器控制，而不是传统的在你的对象内部直接控制。
谁控制谁，控制什么？
传统Java SE程序设计，我们直接在对象内部通过new进行创建对象，是程序主动去创建依赖对象；而IoC是有专门一个容器来创建这些对象，即由Ioc容器来控制对 象的创建。
为何是反转，哪些方面反转了？
因为由容器帮我们查找及注入依赖对象，对象只是被动的接受依赖对象，所以是反转；哪些方面反转了？依赖对象的获取被反转了。
3）DI—Dependency Injection，即“依赖注入”
那么DI是如何实现的呢？ Java 1.3之后一个重要特征是反射（reflection），它允许程序在运行的时候动态的生成对象、执行对象的方法、改变对象的属性，spring就是通过反射来实现注入的。
4)AOP
面向切面编程，通过预编译方式和运行期动态代理实现程序功能的统一维护的一种技术。AOP是OOP(面向对象)的延续，是软件开发中的一个热点，也是Spring框架中的一个重要内容，是函数式编程的一种衍生范型。
5）Spring Bean类
Spring Bean是事物处理组件类和实体类（POJO）对象的总称，Spring Bean被Spring IOC容器初始化，装配和管理。
Spring框架几种创建bean的方式：
XML显示配置Bean
<bean class= "man.BigMan" />
通过Java的注解机制来实现的
@Component，@Service和@Constroller
通过Java配置Bean，也是通过注解来实现的，首先要通过@Configuration注解声明一个配置类，该类中应该包含在spring中创建Bean的所有细节，急用@Configuration加上@Bean去注册一个bean对象。



## 42.springboot整合Hibernate

由于Spring boot默认已经集成了Hibernate, 所在我们只需在pom.xml引用jpa及mysql连接库.
<dependency>  
  <groupId>org.springframework.boot</groupId>  
  <artifactId>spring-boot-starter-data-jpa</artifactId>  
</dependency>  

<dependency>  
  <groupId>mysql</groupId>  
  <artifactId>mysql-connector-java</artifactId>  
</dependency>

spring.jpa.database = MYSQL
//数据库为Mysql
spring.jpa.show-sql = true
//sql语句会输出到控制台
spring.jpa.hibernate.ddl-auto = update
//spring.jpa.properties.hibernate.hbm2ddl.auto=create-drop
//自动创建|更新|验证数据库表结构。如果不是此方面的需求建议设置为"none"，update表示加载hibernate自动更新数据库结构(最常用)。
spring.jpa.hibernate.naming-strategy = org.hibernate.cfg.DefaultNamingStrategy
spring.jpa.properties.hibernate.dialect = org.hibernate.dialect.MySQL5InnoDBDialect
//在将它们添加到实体管理器之前删除

Hibernate的Annotation注解
一、声明实体
@Entity
对实体注释。任何Hibernate映射对象都要有这个注释
@Table
声明此对象映射到数据库的数据表，通过他可以为实体指定表(Table)，目录(Catalog)和Schema的名字。该注释不是必须的，如果没有则系统使用默认值（实体的短类名）
@Version
该注释可用于在实体Bean中添加乐观锁支持。
二、声明主键
@Id
声明此属性为主键。该属性值可以通过自身创建，但是Hibernate推荐通过Hibernate生成。
@GeneratedValue
指定主键的生成策略。有如下四个值
TABLE：使用表保存id值
IDENTITY：identity column
SEQUENCR：sequence排序
AUTO：根据数据库的不同，使用上面三种
三、声明普通属性
@Column
声明该属性与数据库字段的映射关系

@Query与@Modifying执行更新操作
问题：InstantiationException: No default constructor for entity
注意：hibernate在创建实体类对象时需要一个默认的无参构造函数



## 43.整合WebSocket

<!-- WebSocket依赖，移除Tomcat容器 -->
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-websocket</artifactId>
      <exclusions>
        <exclusion>
          <groupId>org.springframework.boot</groupId>
          <artifactId>spring-boot-starter-tomcat</artifactId>
        </exclusion>
      </exclusions>
    </dependency>

基于Stomp的websocket
//注解开启使用STOMP协议来传输基于代理(message broker)的消息,这时控制器支持使用@MessageMapping,就像使用@RequestMapping一样
编写配置文件StompWebsocket:
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig extends AbstractWebSocketMessageBrokerConfigurer{
@Override
        public void registerStompEndpoints(StompEndpointRegistry registry) {//注册STOMP协议的节点(endpoint),并映射指定的url
            //注册一个STOMP的endpoint,并指定使用SockJS协议
            registry.addEndpoint("/endpointZwj").withSockJS();

        }
    
        @Override
        public void configureMessageBroker(MessageBrokerRegistry registry) {//配置消息代理(Message Broker)
            //广播式应配置一个/topic消息代理
            registry.enableSimpleBroker("/topic");
    
        }
}
写Controller
@Controller
    public class WebSocketController {

        @MessageMapping("/welcome") //当浏览器向服务端发送请求时,通过@MessageMapping映射/welcome这个地址,类似于@ResponseMapping
        @SendTo("/topic/getResponse")//当服务器有消息时,会对订阅了@SendTo中的路径的浏览器发送消息
        public AricResponse say(AricMessage message) {
            try {
                //睡眠1秒
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return new AricResponse("welcome," + message.getName() + "!");
        }
    }

客户端使用：
var socket = new SockJS('/endpointAric'); //连接SockJS的endpoint名称为"endpointWisely"
stompClient = Stomp.over(socket);//使用STMOP子协议的WebSocket客户端
stompClient.connect({},function(frame){//连接WebSocket服务端})



## 44.配置swagger

​	<dependency>
​            <groupId>io.springfox</groupId>
​            <artifactId>springfox-swagger2</artifactId>
​            <version>2.8.0</version>
​        </dependency>
​        <dependency>
​            <groupId>io.springfox</groupId>
​            <artifactId>springfox-swagger-ui</artifactId>
​            <version>2.8.0</version>
​        </dependency>
创建config文件夹创建Swagger类：
@Configuration
@EnableSwagger2
public class Swagger {
​    @Bean
​    public Docket createRestApi() {
​        return new Docket(DocumentationType.SWAGGER_2)
​                .apiInfo(apiInfo())
​                .select()
​                .apis(RequestHandlerSelectors.basePackage("com.hibernate.hibernatedemo.controller"))
​                .paths(PathSelectors.any())
​                .build();
​    }
​    private ApiInfo apiInfo() {
​        return new ApiInfoBuilder()
​                .title("springboot利用swagger构建api文档")
​                .description("简单优雅的restfun风格，http://blog.csdn.net/wu_zi")
​                .termsOfServiceUrl("http://blog.csdn.net/wu_zi")
​                .version("1.0")
​                .build();
​    }
}
创建Controller做测试
@RestController
@RequestMapping("/test")
public class hello {
​    @RequestMapping(value = "hello",method = {RequestMethod.POST})
​    public String helloWorld(){
​        return "helloWorld";
​    }
}
访问：
http://localhost:8080/swagger-ui.html



## 45.lombok

<!-- https://mvnrepository.com/artifact/org.projectlombok/lombok -->
<dependency>
    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
    <version>1.18.6</version>
    <scope>provided</scope>
</dependency>
Lombok能以简单的注解形式来简化java代码，提高开发人员的开发效率.帮助写setter,getter。
@Data
会为类的所有属性自动生成setter/getter、equals、canEqual、hashCode、toString方法，如为final属性，则不会为该属性生成setter方法。
如果觉得@Data太过残暴,可以使用@Getter/@Setter注解，此注解在属性上，可以为相应的属性自动生成Getter/Setter方法:
@Getter @Setter private int age = 10;
@NonNull
该注解用在属性或构造器上，Lombok会生成一个非空的声明，可用于校验参数，能帮助避免空指针。
public NonNullExample(@NonNull Person person) {}
@Cleanup
该注解能帮助我们自动调用close()方法，很大的简化了代码。
@Cleanup InputStream in = new FileInputStream(args[0]);
@EqualsAndHashCode
默认情况下，会使用所有非静态（non-static）和非瞬态（non-transient）属性来生成equals和hasCode，也能通过exclude注解来排除一些属性。
@EqualsAndHashCode(exclude={"id", "shape"})
public class EqualsAndHashCodeExample {}
@ToString
类使用@ToString注解，Lombok会生成一个toString()方法，默认情况下，会输出类名、所有属性（会按照属性定义顺序），用逗号来分割。
@ToString(exclude="id")
public class ToStringExample {}



## 46.spring cloud

1）Spring Cloud的核心功能：
分布式/版本化配置
服务注册和发现
路由
服务和服务之间的调用
负载均衡
断路器
2）经常用的5个常用组件
服务发现——Netflix Eureka
客服端负载均衡——Netflix Ribbon
断路器——Netflix Hystrix
服务网关——Netflix Zuul
分布式配置——Spring Cloud Config
分布式消息传递
3）各组件配置使用运行流程：
1、请求统一通过API网关（Zuul）来访问内部服务.
2、网关接收到请求后，从注册中心（Eureka）获取可用服务
3、由Ribbon进行均衡负载后，分发到后端具体实例
4、微服务之间通过Feign进行通信处理业务
5、Hystrix负责处理服务超时熔断
6、Turbine监控服务间的调用和熔断相关指标

创建springcloud主工程：
创建普通的maven工程即可

### 1、Eureka:

#### 1.1、eureka server

Idea：spring initialir -> cloud discovery -> eureka server 
cap原理c一致性、a可用性、p分区容错性不可同时满足，eureka保证的是ap,zookeeper保证的是cp
添加eureka server工程
在springcloud主工程，右键创建module,创建springboot项目，工程名eureka-server 
在选择依赖时选择web->spring web、spring cloud discovery -> eureka server
在启动类下添加@EnableEurekaServer
配置文件添加：

```
server.port=7001
eureka.instance.hostname=localhost
#是否向注册中心注册自己，服务端不向自己注册自己
eureka.client.register-with-eureka=false
#不检索其它服务
eureka.client.fetch-registry=false
#指定服务注册中心位置，监控页面
eureka.client.service-url.defaultZone = http://${eureka.instance.hostname}:${server.port}/eureka/
```

注意：
spring-cloud-starter-eureka-server是1.5以前的版本依赖；
spring-cloud-starter-netflix-eureka-server是最新版本的依赖（推荐）

启动eureka server工程，访问：http://127.0.0.1:7001/（注册中心）

#### 1.2、服务注册到eureka

创建服务提供者项目，一个springboot项目，server-provider

添加依赖：

```
<properties>        
        <spring-cloud.version>2020.0.2</spring-cloud.version>
    </properties>
<dependencies>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
</dependencies>
<dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.cloud</groupId>
                <artifactId>spring-cloud-dependencies</artifactId>
                <version>${spring-cloud.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
```

启动类添加@EnableEurekaClient

配置文件添加：

```
server.port=8088
# 服务名
spring.application.name=server-provider
# eureka注册证中心
eureka.client.service-url.defaultZone=http://localhost:7001/eureka
```

再访问http://127.0.0.1:7001/，会在application看到一个实例

#### 1.3、高可用集群

1）拷贝eureka-server的application.properties，为application-8761.properties、application-8762.properties

修改配置文件的port分别为8761、8762

修改eureka.instance.hostname分别为localhost8761、localhost8762

2）拷贝EurekaServerApplication为EurekaServerApplication8761、EurekaServerApplication8762，并且在class文件上右键选择create 'EurekaServerApplication...'，在environment的program arguments添加：

--spring.profiles.active=8761，在EurekaServerApplication8762执行同样的操作。

3）修改host,添加

127.0.0.1 localhost8761

127.0.0.1 localhost8762

4）分别启动EurekaServerApplication8761、EurekaServerApplication8762。

页面访问：http://localhost8761:8761、http://localhost8762:8762

5）服务provider修改：

```bash
# eureka注册证中心
eureka.client.service-url.defaultZone=http://localhost8761:8761/eureka,http://localhost8762:8762/eureka
```

6）consumer也修改：

```bash
# eureka注册证中心
eureka.client.service-url.defaultZone=http://localhost8761:8761/eureka,http://localhost8762:8762/eureka
```

#### 1.4、eureka自我保护

默认情况下，**如果Eureka Server在一定时间内（默认90秒）没有接收到某个微服务实例的心跳，Eureka Server将会移除该实例。**但是当网络分区故障发生时，微服务与Eureka Server之间无法正常通信，而微服务本身是正常运行的，此时不应该移除这个微服务，所以引入了自我保护机制。

官方对于自我保护机制的定义：

> 自我保护模式正是一种针对网络异常波动的安全保护措施，使用自我保护模式能使Eureka集群更加的健壮、稳定的运行。

自我保护机制的工作机制是：**如果在15分钟内超过85%的客户端节点都没有正常的心跳，那么Eureka就认为客户端与注册中心出现了网络故障，Eureka Server自动进入自我保护机制**，此时会出现以下几种情况：

1. Eureka Server不再从注册列表中移除因为长时间没收到心跳而应该过期的服务。
2. Eureka Server仍然能够接受新服务的注册和查询请求，但是不会被同步到其它节点上，保证当前节点依然可用。
3. 当网络稳定时，当前Eureka Server新的注册信息会被同步到其它节点中。

因此Eureka Server可以很好的应对因网络故障导致部分节点失联的情况，而不会像ZK那样如果有一半不可用的情况会导致整个集群不可用而变成瘫痪。

`eureka.server.enable-self-preservation` 来`true`打开/`false`禁用自我保护机制，默认打开状态，建议生产环境打开此配置。

注册中心关闭自我保护机制，修改检查失效服务的时间。

```bash
eureka:
  server:
     enable-self-preservation: false
     eviction-interval-timer-in-ms: 3000
```

微服务客户端修改减短服务心跳的时间。

```bash
# 每隔2s，向服务端发送一次心跳，默认30s
eureka.instance.lease-renewal-interval-in-seconds=2

# 告诉服务器，10s内没有给你发心跳，代表我故障了,默认90s
eureka.instance.lease-expiration-duration-in-seconds=10
```

以上配置建议在生产环境使用默认的时间配置。

### 2、Ribbon

ribbon是客户端复制均衡，客户端负载均衡和服务端负载均衡最大的不同点在于上面所提到服务清单所存储的位置。在客户端负载均衡中，所有客户端节点都维护着自己要访问的服务端清单，而这些服务端端清单来自于服务注册中心，比如Eureka服务端。同服务端负载均衡的架构类似，在客户端负载均衡中也需要心跳去维护服务端清单的健康性，默认会创建针对各个服务治理框架的Ribbon自动化整合配置。

在主工程添加module，新建springboot项目作为服务消费者，server-consumer，

添加依赖：spring web、eureka client、ribbon、spring cloud config -> config client

注意：springboot版本，springboot2.4后无法添加ribbon依赖，已经不推荐了，推荐spring-cloud-loadbalancer

ribbon是和restTemplate整合到一起的。

#### 2.1、消费provider

添加依赖：

```xml
<properties>        
        <spring-cloud.version>2020.0.2</spring-cloud.version>
    </properties>
<dependencies>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
</dependencies>
<dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.cloud</groupId>
                <artifactId>spring-cloud-dependencies</artifactId>
                <version>${spring-cloud.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
```

启动类添加@EnableEurekaClient

配置文件添加：

```properties
server.port=8088
# 服务名
spring.application.name=server-consumer
# eureka注册证中心
eureka.client.service-url.defaultZone=http://localhost:7001/eureka
```

添加config目录，添加BeanConfig.class

```java
@Configuration
public class BeanConfig {    
    @LoadBalanced
    @Bean
    public RestTemplate restTemplate() {
        return  new RestTemplate();
    }
}
```

创建controller做测试：

```java
@RestController
public class WebController {
    @Autowired
    private RestTemplate restTemplate;

    @GetMapping("web/hello")
    public String hello () {
        System.out.println("consumer hello");
//        return restTemplate.getForEntity("http://127.0.0.1:8088/hello", String.class).getBody() + "  consumer";
        return restTemplate.getForEntity("http://server-provider/hello", String.class).getBody() + "  consumer";
    }
}
```

在provider工程也添加controller提供给consumer消费：

```java
@RestController
public class HelloWorld {
    @GetMapping("/hello")
    public String hello() {
        return "hello world";
    }
}
```

#### 2.2、负载均衡：

把provider再拷贝一份，在主工程点file->new->module from existing sources,把刚拷贝的项目导入。

修改项目名、端口，服务名spring.application.name保证一样不变。

启动2个provider,

访问http://127.0.0.1:8089/web/hello，接口会轮询（默认策略）访问服务。

修改负载均衡策略：

添加依赖

```xml
<!-- https://mvnrepository.com/artifact/com.netflix.ribbon/ribbon -->
<dependency>
    <groupId>com.netflix.ribbon</groupId>
    <artifactId>ribbon</artifactId>
    <version>2.7.18</version>
    <scope>runtime</scope>
</dependency>
<!-- https://mvnrepository.com/artifact/org.springframework.cloud/spring-cloud-starter-netflix-ribbon -->
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-ribbon</artifactId>
    <version>2.2.8.RELEASE</version>
</dependency>
```

BeanConfig添加：

```java
@Bean
public IRule iRule() {
    return new RandomRule();
}
```

#### 2.3、升级

升级问题，新版本Spring Cloud的ribbon推荐用 `spring-cloud-loadbalancer`替代

关闭ribbon：

```properties
spring.cloud.loadbalancer.ribbon.enabled=false
```

因为新版本Spring Cloud的ribbon用 `spring-cloud-loadbalancer`替代了

#### 2.4、restTemplate

GET:

getForEntity

getForObject直接拿body

POST:

postForObject

postForLocation

postForEntity

PUT:

restTemplate.put

DELETE:

restTemplate.delete



### 3、Hystrix

**一个服务失败，导致整条链路的服务都失败的情形，我们称之为服务雪崩**。

解决方案
1） 应用扩容（扩大服务器承受力）

加机器
升级硬件
2）流量控制（超出限定流量，返回类似重试页面让用户稍后再试）

限流
关闭重试
3） 缓存

将用户可能访问的数据大量的放入缓存中，减少访问数据库的请求。

4）服务降级

服务接口拒绝服务
页面拒绝服务
延迟持久化
随机拒绝服务
5） 服务熔断

#### 3.1| 使用hystrix

在server-consumer添加依赖：

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-hystrix</artifactId>
</dependency>
```

在启动类添加，@EnableHystrix

@SpringCloudApplication可以替换以下3个注解

@SpringBootApplication
@EnableEurekaClient
@EnableHystrix

controller添加：

```java
@GetMapping("/web/hystrix")
@HystrixCommand(fallbackMethod = "error")
public String hystrix () {
    System.out.println("consumer hystrix hello");
    return restTemplate.getForEntity("http://server-provider/hello", String.class).getBody() + "  consumer hystrix";
}

public String error() {
    return "error";
}
```

3.2、设置超时熔断时间

默认1s，超过1s就熔断，需要调长些。

```java
@HystrixProperty(name = "execution.isolation.thread.timeoutInMilliseconds", value = "1500")
```

#### 3.3、服务降级

当服务器压力剧增的情况下，根据实际业务情况及流量，对一些服务和页面有策略的不处理或换种简单的方式处理，从而释放服务器资源以保证核心交易正常运作或高效运作。

#### 3.4、异常

```java
public String error(Throwable throwable) {
    System.out.println( "异常：" + throwable.getMessage());
    return "error";
}
```

忽略异常：

```java
@HystrixCommand(fallbackMethod = "error", ignoreExceptions = Exception.class)
```

自定义hystrix：

创建MyHystrixCommand类:

```java
public class MyHystrixCommand extends HystrixCommand<String > {
    private RestTemplate restTemplate;

    public MyHystrixCommand(Setter setter, RestTemplate restTemplate) {
        super(setter);
        this.restTemplate = restTemplate;
    }

    @Override
    protected String run() throws Exception {
        return null;
    }

    @Override
    protected String getFallback() {
        Throwable throwable = super.getExecutionException();
        return super.getFallback();
    }
}
```

```java
@GetMapping("/web/my_hystrix")
public String myHystrix() {
    MyHystrixCommand myHystrixCommand = new MyHystrixCommand(com.netflix.hystrix.HystrixCommand.Setter.withGroupKey(HystrixCommandGroupKey.Factory.asKey("")), restTemplate);
    String str = myHystrixCommand.execute();
    
    Future<String> future =  myHystrixCommand.queue(); // 异步执行
    String str = future.get();
    
    return "hello";
}
```

3.5、仪表盘 dashboard

新建springboot工程, server-hystrix-dashboard，添加依赖：

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-hystrix-dashboard</artifactId>
</dependency>
```

修改配置文件端口为3721

http://127.0.0.1:3721/hystrix

在server-consumer项目，添加依赖：

```xml
<!--  spring boot 提供健康检查      -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

配置文件添加：

```properties
management.endpoints.web.exposure.include=*
```

访问：http://127.0.0.1:8089/actuator/hystrix.stream（要先访问server-consumer的其它接口）

server-hystrix-dashboard配置文件：

```properties
hystrix.dashboard.proxy-stream-allow-list=*
```

在http://127.0.0.1:3721/hystrix添加监控连接，取个名字。

3.5、升级

推荐Resilience4j代替hystrix

### 4、feign

Spring Cloud Feign是基于Netflix feign实现，整合了Spring Cloud Ribbon和Spring Cloud Hystrix

OpenFeign是Spring Cloud 在Feign的基础上支持了Spring MVC的注解，如`@RequesMapping`。

在主工程添加module,创建springboot工程, server-feign,添加依赖：

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

启动类添加：@EnableFeignClients

#### 4.1、声明服务：

创建service目录，添加接口：

```java
/**
 * 绑定服务名，大小写都可以
 **/
@FeignClient("server-provider")
public interface HelloService {
    @RequestMapping("/hello")
    public String hello();
}
```

创建controller：

```java
@RestController
public class Hello {
    @Autowired
    HelloService helloService;

    @RequestMapping("/web/feign/hello")
    public String hello() {
        return helloService.hello();
    }
}
```

配置文件添加：

```properties
server.port=9000

# 服务名
spring.application.name=server-feign
# eureka注册证中心
eureka.client.service-url.defaultZone=http://localhost8761:8761/eureka,http://localhost8762:8762/eureka
```

#### 4.2、负载均衡

使用在接口使用RequestMapping，默认是负载均衡的。

#### 4.3、熔断

配置文件

```properties
feign.hystrix.enabled=true
hystrix.command.default.execution.isolation.thread.timeoutInMilliseconds=2000
```

创建MyFallBack.class

```java
@Component
public class MyFallBack implements HelloService {
    @Override
    public String hello() {
        return "远程服务不可用";
    }
}
```

```java
@FeignClient(value = "server-provider", fallback = MyFallBack.class)
@Component
public interface HelloService {
    @RequestMapping("/hello")
    public String hello();
}
```

升级：openfeign不再包含hystrix



### 5、zuul

　Zuul是spring cloud中的微服务网关。网关： 是一个网络整体系统中的前置门户入口。请求首先通过网关，进行路径的路由，定位到具体的服务节点上。

　　Zuul是一个微服务网关，首先是一个微服务。也是会在Eureka注册中心中进行服务的注册和发现。也是一个网关，请求应该通过Zuul来进行路由。

　　Zuul网关不是必要的。是推荐使用的。

　　使用Zuul，一般在微服务数量较多（多于10个）的时候推荐使用，对服务的管理有严格要求的时候推荐使用，当微服务权限要求严格的时候推荐使用。

#### 5.1、使用

创建module，springboot项目，server-zuul

依赖：

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-zuul</artifactId>
</dependency>
```

启动类添加：

```java
@EnableZuulProxy
```

配置类：

```properties
server.port=9010
spring.application.name=server-zuul
# 匹配符号/api-wkcto/**的请求
zuul.routes..api-wkcto.path=/api-wkcto/**
# 路由到feign
zuul.routes.api-wkcto.service-id=server-feign
# eureka注册证中心
eureka.client.service-url.defaultZone=http://localhost8761:8761/eureka,http://localhost8762:8762/eureka
```

启动项目访问：http://127.0.0.1:9010/api-wkcto/web/feign/hello，就会去访问server-feign的/web/feign/hello接口

#### 5.2、请求过滤

创建AuthFilter.class

```java
@Component
public class AuthFilter extends ZuulFilter {
    @Override
    public String filterType() {
        return "pre";
    }

    @Override
    public int filterOrder() {
        // 执行顺序
        return 0;
    }

    @Override
    public boolean shouldFilter() {
        // 过滤器是否执行
        return true;
    }

    @Override
    public Object run() throws ZuulException {
        RequestContext ctx = RequestContext.getCurrentContext();
        HttpServletRequest request = ctx.getRequest();
        String token = request.getParameter("token");
        if (token == null) {
            ctx.setSendZuulResponse(false);
            ctx.setResponseStatusCode(401);
            ctx.addZuulResponseHeader("content-type", "text/html;charset=utf-8");
            ctx.setResponseBody("非法访问");
        }
        return null;
    }
}

```

#### 5.3、路由规则

```properties
# 忽略映射规则
zuul.ignored-services=server-provider,server-consumer
```

忽略后无法使用http://127.0.0.1:9010/server-provider/server/hello?token=123访问。

```properties
#忽略接口路径
zuul.ignored-patterns=/**/hello/**
```

路由添加前缀：

```properties
# 路由网格前缀
zuul.prefix=/myApi
```

访问需要添加/myApi：http://127.0.0.1:9010/myApi/api-wkcto/web/feign/hello?token=123

*匹配任意字符除了/， **匹配任意字符包括路径/

#### 5.4、业务处理

可以让请求到达网关后，再转发给自己本身，由api网关自己处理。

创建controller：

```java
@RestController
public class Hello {
    @GetMapping("/api/local")
    public String hello() {
        System.out.println("zuul 处理页面hello");
        return "hello zuul";
    }
}
```

配置文件：

```properties
# 路由规则
zuul.routes.gateway.path=/gateway/**
zuul.routes.gateway.url=forward:/api/local
```

访问http://127.0.0.1:9010/myApi/gateway?token=123

#### 5.5、异常过滤器

统一处理异常：

```properties
# 禁用默认错误过滤器
zuul.SendErrorFilter.error.disable=true
```

创建ErrorFilter.class

```java
@Component
public class ErrorFilter extends ZuulFilter {
    @Override
    public String filterType() {
        return "error";
    }

    @Override
    public int filterOrder() {
        return 1;
    }

    @Override
    public boolean shouldFilter() {
        return true;
    }

    @Override
    public Object run() throws ZuulException {
        RequestContext ctx = RequestContext.getCurrentContext();
        ZuulException exception = (ZuulException) ctx.getThrowable();
        System.out.println("进入异常");
        exception.printStackTrace();
        HttpServletResponse response = ctx.getResponse();
        ctx.setResponseStatusCode(exception.nStatusCode);
        PrintWriter writer = null;
        try {
            writer = response.getWriter();
            writer.println("code:" + exception.nStatusCode + ",message:" + exception.getMessage());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (writer != null) {
                writer.close();
            }
        }

        return null;
    }
}
```

测试：我们在上面的AuthFilter.class的run里面添加一行int a = 10 / 0;制造错误。

重启访问：http://127.0.0.1:9010/myApi/api-wkcto/web/feign/hello?token=123会看到错误

方法2：

创建ErrorHandlerController.class

```java
@RestController
public class ErrorHandlerController implements ErrorController {
    @Override
    public String getErrorPath() {
        return "/error";
    }
    @GetMapping("/error")
    public Object error () {
        RequestContext ctx = RequestContext.getCurrentContext();
        ZuulException exception = (ZuulException) ctx.getThrowable();
        return exception.nStatusCode + "--" + exception.getMessage();
    }
}
```

测试：把之前的ErrorFilter注释掉，重启。

#### 5.6、升级

spring-cloud-gateway取代zuul 



### 6、spring cloud config

Spring Cloud Config项目是一个解决分布式系统的配置管理方案。它包含了Client和Server两个部分，server提供配置文件的存储、以接口的形式将配置文件的内容提供出去，client通过接口获取数据、并依据此数据初始化自己的应用。

#### 6.1、spring cloud config服务端

创建springboot项目,spring-cloud-config，依赖

```xml
<properties>
    <java.version>1.8</java.version>
    <spring-cloud.version>2020.0.2</spring-cloud.version>
</properties>
<dependencies>
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-config-server</artifactId>
    </dependency>
</dependencies>
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>${spring-cloud.version}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

启动类添加：@EnableConfigServer

配置文件：

```properties
server.port=3700
spring.application.name=spring-cloud-config
spring.cloud.config.server.git.uri=https://gitee.com/zmrzwj/spring-cloud-config-study.git
spring.cloud.config.server.git.search-paths=config-center
spring.cloud.config.server.git.username=1790373371@qq.com
spring.cloud.config.server.git.password=xxx
```

本地创建目录wkcto,里面再创建config-center,添加

application.properties

application-dev.properties

application-test.properties

application-online.properties

http://127.0.0.1:3700/application/dev/master

#### 6.2、spring cloud config客户端

创建springboot项目：

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <!--  2020.X.X版本官方重构了bootstrap引导配置的加载方式，需要添加以下依赖：-->
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-starter-bootstrap</artifactId>
    </dependency>
    
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-starter-config</artifactId>
    </dependency>   
</dependencies>
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>${spring-cloud.version}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

创建bootstrap.properties配置文件：

```properties
server.port=3701
spring.application.name=application
spring.cloud.config.profile=dev
spring.cloud.config.label=master
spring.cloud.config.uri=http://localhost:3700/
```

创建controller:

```java
@RestController
public class Hello {
    @Value("${url}")
    private String url;

    @RequestMapping("/cloud/url")
    public String url () {
        return url;
    }

    @Autowired
    private Environment env;

    @RequestMapping("/cloud/url2")
    public String url2 () {
        return env.getProperty("url");
    }
}
```

启动项目，访问：http://127.0.0.1:3701/cloud/url 或 http://127.0.0.1:3701/cloud/url2

#### 6.3、安全保护

在服务端添加依赖：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

配置文件添加：

```properties
spring.security.user.name=sccddw
spring.security.user.password=sccddw123456
```

客户端bootstrap.properties配置文件添加：

```properties
spring.cloud.config.username=sccddw
spring.cloud.config.password=sccddw123456
```

7、nacos

替换eureka。





## 47.tomcat部署多个spring boot项目

spring.jmx.default-domain=proj01
spring.jmx.default-domain=proj02

server.xml

```
<Context path="/ehome" docBase="/usr/local/tomcat/webapps/ehome" reloadable="true" />
<Context path="/ehomewx" docBase="/usr/local/tomcat/webapps/ehomewx" reloadable="true" />
```



## 48.Nexus搭建maven私服

下载：
https://www.sonatype.com/download-oss-sonatype
nexus仓库管理器，分为两个版本，Nexus Repository Manager OSS 和 Nexus Repository Manager Pro。前者可以免费使用，相比后者，功能缺少一些，但是不影响我们搭建maven私服。
下载nexus-3.16.1-02-unix.tar.gz
解压到/usr/local
vim bin/nexus.rc
run_as_user="root" #运行用户为root

vim bin/nexus
INSTALL4J_JAVA_HOME_OVERRIDE=/usr/local/java

vim etc/nexus-default.properties
application-port=10000 #运行端口为10000

添加环境变量：
export NEXUS_HOME=/usr/local/nexus
export PATH=$PATH:$NEXUS_HOME/bin

开机启动：
方法1：
cp /usr/local/nexus/bin/nexus /etc/init.d/nexus
cd /etc/init.d
chkconfig --add nexus

设置在3、4、5这3个系统运行级别的时候自动开启nexus服务

sudo chkconfig --levels 345 nexus on

方法2：
使用systemd(CentOS-7推荐使用)
vim /etc/systemd/system/nexus.service

```shell
[Unit]
Description=nexus service
After=network.target
[Service]
Type=forking
LimitNOFILE=65536
ExecStart=/usr/local/nexus/bin/nexus start
ExecStop=/usr/local/nexus/bin/nexus stop
User=root
Restart=on-abort
[Install]
WantedBy=multi-user.target
```

systemctl daemon-reload

#设置开机启动
systemctl enable nexus.service 
systemctl start nexus.service


service nexus start
登录管理页面：
http://192.168.11.206:10000/
管理员默认账户密码为admin/admin123

使用请看：https://cloud.tencent.com/developer/article/1336556



## 49.异步service

@Service
public class AsynSerivce {
    

    @Async
    public void hello() {
        try {
            Thread.sleep(3);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("处理数据中...");
    }
}

@RestController
public class AsynController {

    @Autowired
    AsynSerivce asynSerivce;
    
    @GetMapping("/hello")
    public String hello() {
        asynSerivce.hello();
        return "success";
    }
}
当我们访问时，发现我们睡眠的3秒没有起作用，而是直接就执行了这个方法，不会被阻塞在这里异步处理hello请求



## 50.@Component, @Repository, @Service的区别

在Spring2.0之前的版本中，@Repository注解可以标记在任何的类上，用来表明该类是用来执行与数据库相关的操作（即dao对象），并支持自动处理数据库操作产生的异常
在Spring2.5版本中，引入了更多的Spring类注解：@Component,@Service,@Controller。@Component是一个通用的Spring容器管理的单例bean组件。
而@Repository, @Service, @Controller就是针对不同的使用场景所采取的特定功能化的注解组件。
当你的一个类被@Component所注解，那么就意味着同样可以用@Repository, @Service, @Controller来替代它，同时这些注解会具备有更多的功能，而且功能各异。
最后，如果你不知道要在项目的业务层采用@Service还是@Component注解。那么，@Service是一个更好的选择。



## 51.@Configuration

@Configuration用于定义配置类，可替换xml配置文件，被注解的类内部包含有一个或多个被@Bean注解的方法，这些方法将会被AnnotationConfigApplicationContext或AnnotationConfigWebApplicationContext类进行扫描，并用于构建bean定义，初始化Spring容器。
注意：
@Configuration注解的配置类有如下要求：
@Configuration不可以是final类型；
@Configuration不可以是匿名类；
嵌套的configuration必须是静态类。



## 52.HandlerMethod

HandlerMethod是springMVC中用@Controller声明的一个bean及对应的处理方法.
if(!(object instanceof HandlerMethod)){// 如果不是是SpringMVC Controller请求，直接通过
            return true;
        }



## 53.isAnnotationPresent, getAnnotation方法

指定类型的注释存在于此元素上
A.isAnnotationPresent(B.class)；意思就是：注释B是否在此A上。如果在则返回true；不在则返回false。
getAnnotation 如果存在于此元素，则返回该元素注释指定的注释类型，否则返回为null。
java.lang.reflect.Method.getAnnotation(Class <T> annotationClass)方法如果存在这样的注释，则返回指定类型的元素的注释，否则为null。



## 54.自定义注解

import java.lang.annotation.*;

@Target({ElementType.METHOD, ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Token {
    String value() default "";
}
自定义注解类编写的一些规则:
1. Annotation型定义为@interface, 所有的Annotation会自动继承java.lang.Annotation这一接口,并且不能再去继承别的类或是接口.
2. 参数成员只能用public或默认(default)这两个访问权修饰
3. 参数成员只能用基本类型byte,short,char,int,long,float,double,boolean八种基本数据类型和String、Enum、Class、annotations等数据类型,以及这一些类型的数组.
4. 要获取类方法和字段的注解信息，必须通过Java的反射技术来获取 Annotation对象,因为你除此之外没有别的获取注解对象的方法
5. 注解也可以没有定义成员, 不过这样注解就没啥用了

@Documented 注解
    功能：指明修饰的注解，可以被例如javadoc此类的工具文档化，只负责标记，没有成员取值。
	
@Retention 注解
功能：指明修饰的注解的生存周期，即会保留到哪个阶段。
RetentionPolicy的取值包含以下三种：
SOURCE：源码级别保留，编译后即丢弃。
CLASS:编译级别保留，编译后的class文件中存在，在jvm运行时丢弃，这是默认值。
RUNTIME： 运行级别保留，编译后的class文件中存在，在jvm运行时保留，可以被反射调用。

@Target 注解
功能：指明了修饰的这个注解的使用范围，即被描述的注解可以用在哪里。
ElementType的取值包含以下几种：
TYPE:类，接口或者枚举
FIELD:域，包含枚举常量
METHOD:方法
PARAMETER:参数
CONSTRUCTOR:构造方法
LOCAL_VARIABLE:局部变量
ANNOTATION_TYPE:注解类型
PACKAGE:包

使用
@Token(value="123")或@Token("123")
使用value以外的属性名时，可以不指定属性名称，但其它都必须使用value1="123"形式赋值。



## 55.获取application配置文件参数的两种方式

application.properties：
zwj.name = zwj
使用@Value方式（常用）
  import org.springframework.beans.factory.annotation.Value;  
  @Value("${test.msg}")  
  private String msg; 
使用ConfigurationProperties
@Component
@ConfigurationProperties(prefix = "person")

 @Value不支持复杂类型
 配置文件的占位符：
 application.properties：
 zwj.age = ${random.int}

 

## 56.多项目（如：金证的）

新建个目录把它们都放里面
idea open这个目录；
然后：file->project structure->Modules-> + ->import module->(选中放到这个目录的项目，每个项目都要这么操作一遍)
（项目相互引用是通过pom.xml里面
		<!-- entity实体 -->
        <dependency>
            <groupId>szkingdom.zhcs.kdum_xzyj</groupId>
            <artifactId>xzyj-entity</artifactId>
            <version>1.0.0-SNAPSHOT</version>
        </dependency>

）



## 57.DO:对应数据库表结构

VO：一般用于前端展示使用
DTO：用于数据传递。（接口入参和接口返回值都可以）
以ssm框架为例：
controller层：
public List<UserVO> getUsers(UserDTO userDto);
Service层：
List<UserDTO> getUsers(UserDTO userDto);
DAO层：
List<UserDTO> getUsers(UserDO userDo);

 

## 58.外部jar包集成

 <dependency>
            <groupId>com.jnrsmcu.sdk.netdevice</groupId>
            <artifactId>netdevice</artifactId>
            <version>2.2.2</version>
            <scope>system</scope>
            <systemPath>${project.basedir}/src/main/resources/lib/RSNetDevice-2.2.2.jar</systemPath>
        </dependency>
 <!-- war包添加外部jar包 -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-war-plugin</artifactId>
                <configuration>
                    <webResources>
                        <resource>
                            <directory>src/main/resources/lib</directory>
                            <targetPath>WEB-INF/lib/</targetPath>
                            <includes>
                                <include>**/*.jar</include>
                            </includes>
                        </resource>
                    </webResources>
                </configuration>
            </plugin>

  

  

##  59.@autowired注解原理

 getFields(),获取当前Class所表示类中所有的public的字段
 public Field[] getDeclaredFields():获取当前Class所表示类中所有的字段,不包括继承的字段.
 Method getMethod(String name, Class<?>... parameterTypes)  
返回一个 Method 对象，它反映此 Class 对象所表示的类或接口的指定公共成员方法
class.getMethods()
该方法是获取本类以及父类或者父接口中所有的公共方法

自动装配过程：
1、根据Class对象，通过反射获取所有的Field和```Method````对象
2、通过反射获取Field和Method上的注解，并判断是否有@Autowired和@Value注解（使用到了spring的ReflectionUtils，）
3、将注解了@Autowired和@Value的Field和Method封装成AutowiredFieldElement和AutowiredMethodElement对象，等待下一步的自动装配。
循环处理父类需要自动装配的元素
4、将需要自动装配的元素封装成InjectionMetadata对象，最后合并到Bean定义的externallyManagedConfigMembers属性中
5、用AutowiredFieldElement或AutowiredMethodElement的inject方法，唤起后续步骤
6、通过调用容器的getBean()方法找到需要注入的源数据Bean
7、通过反射将找到的源数据Bean注入到目标Bean中



## 60.redis集成

<!--redis-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-redis</artifactId>
            <version>2.2.5.RELEASE</version>
        </dependency>
        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-pool2</artifactId>
        </dependency>
SpringBoot2.0默认采用Lettuce客户端来连接Redis服务端的
默认是不使用连接池的，只有配置 redis.lettuce.pool下的属性的时候才可以使用到redis连接池
#redis

Redis数据库索引（默认为0）

spring.redis.database=0

Redis服务器地址

spring.redis.host=192.168.11.206
#Redis服务器连接端口
spring.redis.port=6379
#Redis服务器连接密码（默认为空）
spring.redis.password=qwertyui
#连接池最大连接数（使用负值表示没有限制）
spring.redis.lettuce.pool.max-active=8
#连接池最大阻塞等待时间（使用负值表示没有限制）
spring.redis.lettuce.pool.max-wait=10000
#连接池中的最大空闲连接
spring.redis.lettuce.pool.min-idle=3
#连接超时时间（毫秒）
spring.redis.timeout=30000

创建RedisUtil类：
@Component
public class RedisUtil {
    @Autowired
    private StringRedisTemplate stringRedisTemplate;

    @Autowired
    private RedisTemplate redisTemplate;
    
    public String get(String key) {
        String value = (String) stringRedisTemplate.opsForValue().get(key);
        return value;
    }
    
    public void add(String key, String value) {
        stringRedisTemplate.opsForValue().set(key, value);
    }
    
    public Object getObj(String key) {
        return key == null ? null : redisTemplate.opsForValue().get(key);
    }
    
    public boolean addObj(String key, Object value) {
        try {
            redisTemplate.opsForValue().set(key, value);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
为了使用redisTemplate，需要添加配置：
@Configuration
public class RedisConfig {
    @Bean
    @SuppressWarnings("all")
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory factory) {
        RedisTemplate<String, Object> template = new RedisTemplate<String, Object>();
        template.setConnectionFactory(factory);
        Jackson2JsonRedisSerializer jackson2JsonRedisSerializer = new Jackson2JsonRedisSerializer<>(Object.class);
        ObjectMapper om = new ObjectMapper();
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        om.enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL);
        jackson2JsonRedisSerializer.setObjectMapper(om);
        StringRedisSerializer stringRedisSerializer = new StringRedisSerializer();

        // key采用String的序列化方式
        template.setKeySerializer(stringRedisSerializer);
        // hash的key也采用String的序列化方式
        template.setHashKeySerializer(stringRedisSerializer);
        // value序列化方式采用jackson
        template.setValueSerializer(jackson2JsonRedisSerializer);
        // hash的value序列化方式采用jackson
        template.setHashValueSerializer(jackson2JsonRedisSerializer);
        template.afterPropertiesSet();
    
        return template;
    }
}
写一个测试的实体类：
public class RedisBean  {
    String name;
    Integer age;
    public RedisBean() {} // 这是必须的
    public RedisBean(String name, Integer age) {
        this.name = name;
        this.age = age;
    }
}
注：为什么需要空构造函数，redis的这些序列化方式,使用的是无参构造函数进行创建对象set方法进行赋值,
方法中存在有参的构造函数,默认存在的无参构造函数是不存在的(继承自object),必须显示的去重写.

  

  

## 61.MongoDB集成

<!--mongodb-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-mongodb</artifactId>
        </dependency>  
#mogodb
spring.data.mongodb.database=xizang
spring.data.mongodb.host=192.168.11.206
spring.data.mongodb.password=sccddw
spring.data.mongodb.username=sccddw
spring.data.mongodb.port=27017
创建RedisUtil类：
@Component
public class MongoUtil {
    @Autowired
    MongoTemplate mongoTemplate;

    public boolean saveObj(Object obj) {
        try {
            mongoTemplate.save(obj);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    public Object getObjById (String id) {
        Query query = new Query(Criteria.where("_id").is(id));
        return mongoTemplate.findOne(query, MongoBean.class);
    }
    
    public List<MongoBean> findAll () {
        List<MongoBean> list = null;
        try {
            return mongoTemplate.findAll(MongoBean.class);
        } catch (Exception e) {
            e.printStackTrace();
            return list;
        }
    }
}

新建测试类：
@Data
public class MongoBean {
    private String id;
    private String name;
    private String info;
    private Date createTime;

    public MongoBean(String id, String name, String info, Date createTime) {
        this.id = id;
        this.name = name;
        this.info = info;
        this.createTime = createTime;
    }
    
    @Override
    public String toString() {
        return "id=" + id + ",name=" + name + ",info=" + info + ",createTime" + createTime;
    }
}
注：对象的private属性添加几个getter（、setter）才可以在controller返回给前端。 



## 62.es集成

ES支持两种协议  
       HTTP协议，支持的客户端有Jest client和Rest client
       Native Elasticsearch binary协议，也就是Transport client和Node client
Jest client和Rest client区别
      Jest client非官方支持，在ES5.0之前官方提供的客户端只有Transport client、Node client。在5.0之后官方发布Rest client，并大力推荐.
REST Client
官方推荐使用，所以我们采用这个方式，这个分为两个Low Level REST Client和High Level REST Client，Low Level REST Client是早期出的API比较简陋了，
还需要自己去拼写Query DSL，High Level REST Client使用起来更好用，更符合面向对象的感觉.

<!--rest-->
        <dependency>
            <groupId>org.elasticsearch.client</groupId>
            <artifactId>elasticsearch-rest-client</artifactId>
            <version>7.6.2</version>
        </dependency>
        <dependency>
            <groupId>org.elasticsearch.client</groupId>
            <artifactId>elasticsearch-rest-high-level-client</artifactId>
            <version>7.6.2</version>
        </dependency>
		<dependency>
            <groupId>org.elasticsearch</groupId> (这个需要不然单元测试那会报某些类没有)
            <artifactId>elasticsearch</artifactId>
            <version>7.6.2</version>
        </dependency>
application.properties
spring.elasticsearch.rest.uris=http://127.0.0.1:9200
@Resource
private RestClient client;
或
@Resource
private RestHighLevelClient restHighLevelClient;（推荐）

<!--jest-->
        <dependency>
            <groupId>org.elasticsearch</groupId>
            <artifactId>elasticsearch</artifactId>
            <version>6.5.4</version>
        </dependency>
        <dependency>
            <groupId>io.searchbox</groupId>
            <artifactId>jest</artifactId>
            <version>5.3.3</version>
        </dependency>
application.properties
spring.elasticsearch.jest.uris=http://127.0.0.1:9200
@Resource
private JestClient jestClient;
RestHighLevelClient API:
https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high-document-index.html



## 63.单元测试无法使用Autowired的解决办法

1）单元测试使用junit4
<dependency>
    <groupId>junit</groupId>
    <artifactId>junit</artifactId>
    <version>4.12</version>
    <scope>test</scope>
</dependency>
2）测试类添加
@RunWith(SpringRunner.class)
@SpringBootTest
@Test为import org.junit.Test;



## 64.socketIO集成：

SocketIOConfig：
@org.springframework.context.annotation.Configuration
public class SocketIOConfig {
    @Value("${socketIo.host}")
    private String host;

    @Value("${socketIo.port}")
    private Integer port;
    
    @Value("${socketIo.context}")
    private String context;
    
    @Value("${socketIo.pingTimeout}")
    private int pingTimeout;
    
    @Bean
    public SocketIOServer socketIOServer() {
        Configuration configuration = new Configuration();
        configuration.setHostname(host);
        configuration.setPort(port);
        configuration.setContext(context);
        configuration.setPingTimeout(pingTimeout);
    
        return new SocketIOServer(configuration);
    }
}
SocketIOService：
public interface SocketIOService {
    /**
     * 启动服务
     */
    void start();

    /**
     * 停止服务
     */
    void stop();
}
SocketIOServiceImpl：
@Service(value = "SocketIOService")
public class SocketIOServiceImpl implements SocketIOService {
	/**
     * 存放已连接的客户端
     */
    private static Map<String, SocketIOClient> clientMap = new ConcurrentHashMap<>();

    /**
     * 自定义事件`push_data_event`,用于服务端与客户端通信
     */
    private static final String PUSH_DATA_EVENT = "push_data_event";
    // client.sendEvent(PUSH_DATA_EVENT, msgContent); (发送事件)
    
    @Autowired
    private SocketIOServer socketIOServer;
    
    @Resource
    private ConnectEventListenner connectEventListenner;
    
    @Override
    public void start() {
        // 监听客户端连接
        socketIOServer.addConnectListener(client -> {
            log.info("************ 客户端： " + getIpByClient(client) + " 已连接 ************");
            // 自定义事件`connected` -> 与客户端通信  （也可以使用内置事件，如：Socket.EVENT_CONNECT）
            client.sendEvent("connected", "你成功连接上了哦...");
            Map params = getParamsByClient(client);
            log.info(params.toString());
        });
        // 监听客户端断开连接
        socketIOServer.addDisconnectListener(client -> {
            String clientIp = getIpByClient(client);
            log.info(clientIp + " *********************** " + "客户端已断开连接");
            Map params = getParamsByClient(client);
            log.info(params.toString());
        });
        // 自定义事件`client_info_event` -> 监听客户端消息
        socketIOServer.addEventListener(PUSH_DATA_EVENT, String.class, (client, data, ackSender) -> {
            // 客户端推送`client_info_event`事件时，onData接受数据，这里是string类型的json数据，还可以为Byte[],object其他类型
            log.info(PUSH_DATA_EVENT + " *********************** " + "事件触发");
            log.info(data);
            String clientIp = getIpByClient(client);
            log.debug(clientIp + " ************ 客户端：" + data);
        });
        // 启动服务
        socketIOServer.start();
    }
    
    @Override
    public void stop() {
        if (socketIOServer != null) {
            socketIOServer.stop();
            socketIOServer = null;
        }
    }
    
    /**
     * 获取客户端url中的userId参数（这里根据个人需求和客户端对应修改即可）
     *
     * @param client: 客户端
     * @return: java.lang.String
     */
    private Map getParamsByClient(SocketIOClient client) {
        // 获取客户端url参数（这里的userId是唯一标识）
        Map<String, List<String>> params = client.getHandshakeData().getUrlParams();
        if (!CollectionUtils.isEmpty(params)) {
            return params;
        }
        return null;
    }
    
    /**
     * 获取连接的客户端ip地址
     *
     * @param client: 客户端
     * @return: java.lang.String
     */
    private String getIpByClient(SocketIOClient client) {
        String sa = client.getRemoteAddress().toString();
        String clientIp = sa.substring(1, sa.indexOf(":"));
        return clientIp;
    }
}  

使用CommandLineRunner让socket随springboot启动：
@Component
@Slf4j
public class socketCommandLineRunner implements CommandLineRunner {
    @Autowired
    private SocketIOServiceImpl socketIOService;

    @Override
    public void run(String... args) throws Exception {
        /**
         * 当CommandLineRunner中出现不可预期的异常时，会影响主线程，所以这里单独启动一个线程执行
         */
        new Thread(){
            @Override
            public void run(){
                try {
                    log.info("开始启动");
                    socketIOService.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }.start();
    }
}



## 65.添加阿里maven仓库

idea： file->settings->Maven->user settings file(勾选override)
修改.setting.xml:
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
                          https://maven.apache.org/xsd/settings-1.0.0.xsd">

    <mirrors>
        <mirror>
            <id>alimaven</id>
            <name>aliyun maven</name>
            <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
            <mirrorOf>central</mirrorOf>
        </mirror>
    </mirrors>
</settings>



## 66.文件服务集成

private String filePath = "G:\\";
@PostMapping("/upload")
    @ApiOperation(value="单文件上传")
    public String upload(@RequestParam("file") MultipartFile file) {
        if(file.isEmpty()) {
            return "文件为空";
        }
        String fileName = file.getOriginalFilename();
        String suffixName = fileName.substring(fileName.lastIndexOf("."));
        String uuidName = UUID.randomUUID() + suffixName;
        File dest  = new File(filePath + uuidName);

        try {
            file.transferTo(dest);
            log.info("上传成功");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "success";
    }

@PostMapping("/multiUpload")
    @ApiOperation(value="多文件上传")
    @ResponseBody
    public String multiUpload(HttpServletRequest request) {
        List<MultipartFile> files = ((MultipartHttpServletRequest) request).getFiles("file");
        for(int i = 0;i<files.size();i++) {
            MultipartFile file = files.get(i);
            if(file.isEmpty()) {
                return "第" + (i++) + "个文件为空";
            }
            String fileName = file.getOriginalFilename();
            File dest  = new File(filePath + fileName);
            try {
                file.transferTo(dest);
                log.info("上传成功");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return "{code: 1, msg: \"123\"}";
    }	
@GetMapping("/download")
    @ApiOperation(value="文件下载")
    public ResultInfo downloadFile(@RequestParam("fileName") String fileName, HttpServletResponse response) {
        ResultInfo resultInfo = new ResultInfo();
        File file = new File(filePath + fileName);
        try {
            FileInputStream fileInputStream = new FileInputStream(file);
//            response.setContentType("application/force-download");
            response.addHeader("Content-disposition", "attachment;fileName=" + fileName);
            OutputStream outputStream = response.getOutputStream();

            byte[] buf = new byte[1024];
            int len = 0;
            while ((len = fileInputStream.read(buf)) != -1) {
                outputStream.write(buf, 0, len);
            }
            fileInputStream.close();
            resultInfo.setResultCode(ResultCode.SUCCESS);
            resultInfo.setData("success");
            return resultInfo;
        } catch (IOException e) {
            e.printStackTrace();
        }
        resultInfo.setResultCode(ResultCode.FILE_ERROR);
        return resultInfo;
    }
springboot上传文件大小限制：
spring.servlet.multipart.max-file-size=5000MB
spring.servlet.multipart.max-request-size=5000MB



## 67.profiles多环境切换

新建文件
application-206.properties
在application.properties添加：
spring.profiles.active=206
yml文档块写法：
server
	port: 8080
spring:
	profiles:

​		active: dev

server
	port: 8081
spring
	profiles: dev
命令行方式：
java -jar --spring-profiles.active=dev;

idea多环境切换：
<resources>
            <resource>
                <directory>src/main/java</directory>
                <includes>
                    <include>**/*.properties</include>
                    <include>**/*.xml</include>
                </includes>
                <filtering>true</filtering>
            </resource>
            <resource>
                <directory>src/main/resources</directory>
                <filtering>true</filtering>
                <includes>
                    <include>application-${profiles.active}.properties</include>
                    <include>application.properties</include>
                    <include>**/*.xml</include>
                    <include>**/*.properties</include>
                    <include>META-INF/**</include>
                </includes>
            </resource>

            <resource>
                <directory>src/main/resources</directory>
                <filtering>false</filtering>
                <includes>
                    <include>static/**</include>
                    <include>page/**</include>
                </includes>
            </resource>
        </resources>

<profiles>
        <profile>
            <id>dev</id>
            <properties>
                <profiles.active>dev</profiles.active>
            </properties>
			<activation>
                <activeByDefault>true</activeByDefault>
            </activation>
        </profile>

        <profile>
            <id>test</id>
            <properties>
                <profiles.active>test</profiles.active>
            </properties>
        </profile>
    
        <profile>
            <id>prod</id>
            <properties>
                <profiles.active>prod</profiles.active>
            </properties>
        </profile>
    </profiles>

application.properties添加：	
spring.profiles.active=@profiles.active@	
然后你可以在idea右边maven看到的了一列Profiles	



## 68.springboot配置文件加载顺序

-file:./config
-file:./
-classpath:/config/
-classpath:/
各个配置文件是互补的
命令行指定：
java -jar xxx--spring.config.location=G:/application-dev.properties

springboot debug模式：
application.properties:
debug=true
控制台会打印启用的自动配置类
springboot2的application.properties 官方配置 说明文件：
https://docs.spring.io/spring-boot/docs/2.3.0.RELEASE/reference/html/appendix-application-properties.html#common-application-properties



## 69.Autowired与Resource

@Resource的作用相当于@Autowired，只不过@Autowired按byType自动注入，而@Resource默认按 byName自动注入罢了。
@Resource有两个属性是比较重要的，分是name和type，Spring将@Resource注解的name属性解析为bean的名字，而type属性则解析为bean的类型。
所以如果使用name属性，则使用byName的自动注入策略，而使用type属性时则使用byType自动注入策略。如果既不指定name也不指定type属性，这时将通过反射机制使用byName自动注入策略。
(类型就是类UserDao，name就是@Resource(name="baseDao"))
@Resource
如果没有指定name属性，当注解写在字段上时，默认取字段名进行安装名称查找。如果注解写在setter方法上默认取属性名进行装配。
当找不到与名称匹配的bean时才按照类型进行装配。但是需要注意的是，如果name属性一旦指定，就只会按照名称进行装配。
@Qualifier("service")
@Qualifier注解了，qualifier的意思是合格者，通过这个标示，表明了哪个实现类才是我们所需要的
一般@Autowired和@Qualifier一起用，@Resource单独用。当然没有冲突的话@Autowired也可以单独用



## 70.myBatis集成

创建mapper
1）使用注解
@SpringBootApplication上添加@MapperScan("com.sccddw.iot.mapper")指定mapper包批量扫描，就不用每个类加@Mapper
创建mapper目录
创建xxxMpper接口文件：
@Repository
public interface SDSensorTypeMapper extends BaseMapper<SDSensorType> {
    @Select("select * from #{name};")
    List<SDSensorType> findAll(@Param("name") name);
}
创建service目录
创建IxxxService
public interface IxxxService {
    List<SDSensorType> findAll();
}
在service目录中创建impl目录
新建DmServiceImpl
@Service
public class xxxServiceImpl implements IxxxService {
    @Autowired
    xxxMapper xxxMapper;

    public List<Dm> findAll () {
        return xxxMapper.findAll();
    }
}

2）使用新xml

mybatis-plus

#全局配置文件
mybatis-plus.config-locations=classpath:mybatis/mybatis-config.xml
#mybatis-plus.type-aliases-package=com.sccddw.test.entity.vo,com.sccddw.test.entity.db,com.sccddw.test.entity.bean

指定sql映射文件的位置

mybatis-plus.mapper-locations=classpath:mybatis/mapper/*.xml
mybatis-plus.global-config.db-config.id-type=auto
在resources中创建mybatis目录
创建：mybatis-config.xml
在mybatis目录下创建mapper目录：
单个：DrsrsAllocateDepotOrderDetailMapper.xml

${}将传入的数据直接显示生成在sql中,#{}将数据都当成一个字符串



## 71.shiro集成

<!--shiro-->
        <dependency>
            <groupId>org.apache.shiro</groupId>
            <artifactId>shiro-spring</artifactId>
            <version>1.5.3</version>
        </dependency>
shiro功能：
    认证、授权、加密、会话管理



## 72.oauth2

四种模式：
授权码模式（authorization code）
    网上看到通过微信登录，QQ登录等字眼。通过请求获取授权code，response_type:code（授权码模式下springOAuth2规定为code）;
redirect_uri:授权成功后的回调地址，登录成功后会去到redirect_uri并带上code.然后拿code去鉴权服务拿token。
    对于认证中心：需要注册用户，需要注册应用。
简化模式（implicit）

密码模式（resource owner password credentials）
    主要是适用于客户端来获取用户信息的场景。
	与授权码模式不同，不需要认证服务颁发的授权码，只需要用户名和密码以及我们设置的客户端信息就可以了。
	
客户端模式（client credentials）(主要用于api认证，跟用户无关)
提供给第三方一个client_id 和 client_secret,相当于公用的用户名密码,第三方拿着这俩东西去换取令牌,拿着令牌去取数据就可以了. 
基本只需要提供两个接口 ,  一个是获取令牌 , 一个是获取数据
GET /token?grant_type=client_credentials & client_id = {客户端身份ID} &  相当于用户名client_secret = {客户端秘钥}  & 相当于密码
我们后端拿着这个用户名密码进行比对 , 比对成功 ,返回access_token
2. 第三方拿着access_token 获取数据
GET /user?access_token = {令牌} &uid = {用户ID}
后端验证access_token 存在 , 并且没有过期 , 就验证通过 , 返回数据。



## 73.mybatis-plus

### 73.1 @tableid

mybatis主键注解，指名主键字段，不然默认是id.

### 73.2 mybatis分页

配置文件：
public class MybatisPlusConfig {
    /**
     * 分页插件
     */
    @Bean
    public PaginationInterceptor paginationInterceptor() {
        PaginationInterceptor paginationInterceptor = new PaginationInterceptor();
        paginationInterceptor.setDialectType("mysql");
        return paginationInterceptor;
    }

    /**
     * 打印 sql
     */
//    @Bean
//    public PerformanceInterceptor performanceInterceptor() {
//        PerformanceInterceptor performanceInterceptor = new PerformanceInterceptor();
//        //格式化sql语句
//        Properties properties = new Properties();
//        properties.setProperty("format", "true");
//        performanceInterceptor.setProperties(properties);
//        return performanceInterceptor;
//    }
}

QueryWrapper<Device> queryWrapper = new QueryWrapper<>();
// false不查询总条数, current从1开始
Page<Device> page = new Page<>(1, 2, true);
IPage<Device> iPage = deviceMapper.selectPage(page, queryWrapper);
自定义查询：
在Mapper中添加
@Select("select * from device where device_name like concat('%', #{param}, '%') or device_code = #{param}")
IPage<Device> pageQueryDeviceByDeviceCodeOrDeviceNameMybatis(Page<?> page, @Param("param") String param);
使用：
IPage<Device> iDevicePage = deviceMapper.pageQueryDeviceByDeviceCodeOrDeviceNameMybatis(page, "123");

### 73.3 多数据源

<!-- mybatisplus集成 -->
        <dependency>
            <groupId>com.baomidou</groupId>
            <artifactId>mybatis-plus-boot-starter</artifactId>
            <version>3.1.2</version>
        </dependency>
        <dependency>
            <groupId>com.baomidou</groupId>
            <artifactId>dynamic-datasource-spring-boot-starter</artifactId>
            <version>2.5.6</version>
        </dependency>
application.properties添加：

mysql

spring.datasource.dynamic.datasource.master.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.dynamic.datasource.master.url=jdbc:mysql://192.168.11.206:3306/hndl5000?useUnicode=true&characterEncoding=UTF-8&allowMultiQueries=true&serverTimezone=GMT%2B8
spring.datasource.dynamic.datasource.master.username=sccddw
spring.datasource.dynamic.datasource.master.password=sccddw
spring.datasource.dynamic.primary=master

spring.datasource.dynamic.datasource.slave.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.dynamic.datasource.slave.url=jdbc:mysql://192.168.11.206:3306/hndl5000?useUnicode=true&characterEncoding=UTF-8&allowMultiQueries=true&serverTimezone=GMT%2B8
spring.datasource.dynamic.datasource.slave.username=sccddw
spring.datasource.dynamic.datasource.slave.password=sccddw

在service上使用  @DS("slave")使用指定数据源。

### 73.4 自动填充

// insert时填充
@TableField(fill = FieldFill.INSERT)
// update时填充
@TableField(fill = FieldFill.UPDATE)
/**

 * mybatis-plus自动填充配置
 **/
  @Component
  public class MyMetaObjectHandler implements MetaObjectHandler {
    @Override
    public void insertFill(MetaObject metaObject) {
        boolean hasSetter = metaObject.hasSetter("createTime");
        if (hasSetter) {
            setInsertFieldValByName("createTime", LocalDateTime.now(), metaObject);
        }
    }

    @Override
    public void updateFill(MetaObject metaObject) {
        boolean hasSetter = metaObject.hasSetter("updateTime");
        if (hasSetter) {
            setUpdateFieldValByName("updateTime", LocalDateTime.now(), metaObject);
        }
    }
  }

### 73.5 乐观锁

乐观锁：总是假设最好的情况，每次去读数据的时候都认为别人不会修改，所以不会上锁。
操作逻辑：
每次查询时，查出带有version的数据记录，更新数据时，判断数据库里对应id的记录的version是否和查出的version相同。
若相同，则更新数据并把版本号+1；若不同，则说明，该数据发送并发，
被别的线程使用了，进行递归操作，再次执行递归方法，知道成功更新数据为止。
所以乐观锁需要反复尝试更新某一个变量，如果又一直更新不成功，循环往复，会给CPU带来很大的压力。
为了尽可能避免更新失败，可以合理调整重试次数（阿里巴巴开发手册规定重试次数不低于三次）。
MybatisPlus乐观锁：
MybatisPlusConfig中添加：
// 乐观锁插件配置
    @Bean
    public OptimisticLockerInterceptor optimisticLockerInterceptor() {
        return new OptimisticLockerInterceptor();
    }
	

	在表里添加version字段。
	实体类添加：
	// 版本
	@Version
	private Integer version;
使用：
int id = 100;
int version = 2;
User u = new User();
u.setId(id);
u.setVersion(version);
userService.updateById(u);
	

### 73.6 性能分析插件

MybatisPlusConfig中添加：
/**
     * 打印 sql, 和运行时间
     */
    @Bean
    @Profile({"dev", "test"}) // 在开发环境和测试环境运行，vm参数设置-Dspring.profiles.active=dev
    public PerformanceInterceptor performanceInterceptor() {
        PerformanceInterceptor performanceInterceptor = new PerformanceInterceptor();
        //格式化sql语句
        Properties properties = new Properties();
        properties.setProperty("format", "true");
        performanceInterceptor.setProperties(properties);
		performanceInterceptor.setMaxTime(5); // sql运行的最大时间，超过这个时间就不运行了。
        return performanceInterceptor;
    }

### 73.7 动态表

MybatisPlusConfig中添加：
	/**
     * 分页插件
     */
    @Bean
    public PaginationInterceptor paginationInterceptor() {
        PaginationInterceptor paginationInterceptor = new PaginationInterceptor();
        paginationInterceptor.setDialectType("mysql");

        List<ISqlParser> sqlParsers = new ArrayList<>();
    
        DynamicTableNameParser dynamicTableNameParser = new DynamicTableNameParser();
        Map<String, ITableNameHandler> tableNameHandlerMap = new HashMap<>();
        tableNameHandlerMap.put("sensor_data", new ITableNameHandler() {
            @Override
            public String dynamicTableName(MetaObject metaObject, String sql, String tableName) {
                return myTableName.get();
            }
        });
        dynamicTableNameParser.setTableNameHandlerMap(tableNameHandlerMap);
        sqlParsers.add(dynamicTableNameParser);
        paginationInterceptor.setSqlParserList(sqlParsers);
    
        return paginationInterceptor;
    }
使用：
MybatisPlusConfig.myTableName.set("sensor_e_20201014"); //这里会把sensor_data表替换成sensor_e_20201014表达到动态表名的目的
sensorDataMapper.selectById(1);

### 73.8 自定义方法

创建DeleteAllMethod.class
public class DeleteAllMethod extends AbstractMethod {
    @Override
    public MappedStatement injectMappedStatement(Class<?> mapperClass, Class<?> modelClass, TableInfo tableInfo) {
        String sql = "delete from " + tableInfo.getTableName();
        String method = "deleteAll";
        SqlSource sqlSource = languageDriver.createSqlSource(configuration, sql, modelClass);

        return addDeleteMappedStatement(mapperClass,method, sqlSource);
    }
}
创建MySqlInjector.class
@Component
public class MySqlInjector extends DefaultSqlInjector {
    @Override
    public List<AbstractMethod> getMethodList(Class<?> mapperClass) {
        List<AbstractMethod> methodList = super.getMethodList(mapperClass);
        methodList.add(new DeleteAllMethod());
        return methodList;
    }
}
mapper中：
public interface MyMapper<T> extends BaseMapper<T> {
    int deleteAll();  
}

选装件
MySqlInjector.class：
@Component
public class MySqlInjector extends DefaultSqlInjector {
    @Override
    public List<AbstractMethod> getMethodList(Class<?> mapperClass) {
        List<AbstractMethod> methodList = super.getMethodList(mapperClass);
        methodList.add(new DeleteAllMethod());

        methodList.add(new InsertBatchSomeColumn(t -> !t.isLogicDelete() && !t.getColumn().equals("age"))); // 把逻辑删除的字段和age字段都排除在数据里
        methodList.add(new LogicDeleteByIdWithFill()); // 在逻辑删除时修改些数据
        methodList.add(new AlwaysUpdateSomeColumnById(t -> !t.getColumn().equals("name"))); // name不进行更新
    
        return methodList;
    }
}
mapper中：
public interface MyMapper<T> extends BaseMapper<T> {   // 自定义mapper，在需要使用方法的地方继承此mapper就行
    int insertBatchSomeColumn(List<T> list);
    int deleteByIdWithFill(T entity);
    int alwaysUpdateSomeColumnById(T entity);
}
（对于LogicDeleteByIdWithFill
// 要在逻辑删除时修改此项，需要在字段上加@TableField(fill = FieldFill.UPDATE)
    @TableField(fill = FieldFill.UPDATE)
	private String deviceCode = null;
）

### 73.9 逻辑删除

在表添加字段deleted用于标明逻辑删除，默认0-逻辑未删除值，1-逻辑已删除值，也可修改配置文件添加
#mybatis-plus
mybatis-plus.global-config.db-config.logic-not-delete-value=0
mybatis-plus.global-config.db-config.logic-delete-value=1

对于小于3.1.0版本的在MybatisPlusConfig添加，大于3.1.0的可以忽略：
@Bean
    public ISqlInjector sqlInjector() {
        return new LogicSqlInjector();
    }

对实体添加注解TableLogic：
 @TableLogic
 @TableField(select = false) // 查询的时候不显示，因为deleted字段对用户应该透明
 private Integer deleted;
使用：
sensorDataMapper.deleteById(1); // 他就会将这行deleted置为1，但不会在数据库真的删除，对查询数据是一样的。
（注意：自定义语句要自己添加deleted筛选条件）

### 73.10 AR模式

使在对象上可以直接使用数据库增、删、改、查方法
实体类继承Model：
public class SensorData extends Model<SensorData>
使用：
SensorData sensorData = new SensorData();
sensorData.setId(BigInteger.valueOf(1));
sensorData.insertOrUpdate(); // 插入数据有就更新



## 74.@Transactional

74.1 @Transactional 是声明式事务管理 编程中使用的注解
74.2 @Transactional 注解的属性信息
属性名	说明
name	当在配置文件中有多个 TransactionManager , 可以用该属性指定选择哪个事务管理器。
propagation	事务的传播行为，默认值为 REQUIRED。
isolation	事务的隔离度，默认值采用 DEFAULT即数据库的默认隔离级别。
timeout	事务的超时时间，默认值为-1。如果超过该时间限制但事务还没有完成，则自动回滚事务。
read-only	指定事务是否为只读事务，默认值为 false；为了忽略那些不需要事务的方法，比如读取数据，可以设置 read-only 为 true，无法执行增删改。
rollback-for	用于指定能够触发事务回滚的异常类型，如果有多个异常类型需要指定，各类型之间可以通过逗号分隔。
				Spring默认情况下会对(RuntimeException)及其子类来进行回滚,在遇见Exception及其子类的时候则不会进行回滚操作.
no-rollback- for	抛出 no-rollback-for 指定的异常类型，不回滚事务。

propagation
1. TransactionDefinition.PROPAGATION_REQUIRED：
   如果当前存在事务，则加入该事务；如果当前没有事务，则创建一个新的事务。这是默认值。
2. TransactionDefinition.PROPAGATION_REQUIRES_NEW：
   创建一个新的事务，如果当前存在事务，则把当前事务挂起。
3. TransactionDefinition.PROPAGATION_SUPPORTS：
   如果当前存在事务，则加入该事务；如果当前没有事务，则以非事务的方式继续运行。
4. TransactionDefinition.PROPAGATION_NOT_SUPPORTED：
   以非事务方式运行，如果当前存在事务，则把当前事务挂起。
5. TransactionDefinition.PROPAGATION_NEVER：
   以非事务方式运行，如果当前存在事务，则抛出异常。
6. TransactionDefinition.PROPAGATION_MANDATORY：
   如果当前存在事务，则加入该事务；如果当前没有事务，则抛出异常。
7. TransactionDefinition.PROPAGATION_NESTED：
   如果当前存在事务，则创建一个事务作为当前事务的嵌套事务来运行；
   如果当前没有事务，则该取值等价于TransactionDefinition.PROPAGATION_REQUIRED。

74.3 @Transactional 实质是使用了 JDBC 的事务来进行事务控制的
@Transactional 基于 Spring 的动态代理的机制
1) 事务开始时，通过AOP机制，生成一个代理connection对象，
   并将其放入 DataSource 实例的某个与 DataSourceTransactionManager 相关的某处容器中。
   在接下来的整个事务中，客户代码都应该使用该 connection 连接数据库，
   执行所有数据库命令。
   [不使用该 connection 连接数据库执行的数据库命令，在本事务回滚的时候得不到回滚]
  （物理连接 connection 逻辑上新建一个会话session；
   DataSource 与 TransactionManager 配置相同的数据源）

2) 事务结束时，回滚在第1步骤中得到的代理 connection 对象上执行的数据库命令，
   然后关闭该代理 connection 对象。
  （事务结束后，回滚操作不会对已执行完毕的SQL操作命令起作用）

74.4 一般用在serviceImpl层上，也可在controller里用。
74.5 springboot自动配置事务开启事务，入库类不用任何添加注解.

74.6 失效场景：
1.@Transactional类内部方法调用不会生效，想类内部方法调用可以正常使用事务，使用AopContext.currentProxy()来获取代理类再调用。
因为：AOP 代理下，只有目标方法由外部调用，目标方法才由 Spring 生成的代理对象来管理。
2.@Transaction应用在非public修饰的方法上
3.注解属性propagation设置错误
以下三种情况，事务将不会进行回滚
TransactionDefinition.PROPAGATION_SUPPORTS：
TransactionDefinition.PROPAGATION_NOT_SUPPORTED
TransactionDefinition.PROPAGATION_NEVER
4.注解属性rollbackFor设置错误
rollbackFor可以指定能够触发事务回滚的异常类型，Spring默认抛出未检出unchecked的异常（继承RunTimeException的异常）或者Error才会回滚事务，
其他异常不会触发事务，如果事务中抛出其他异常类型，需要指定rollbackFor属性
5.异常被catch，导致@Transaction失效
6.数据库存储引擎不支持，比如MyISAM

74.7 多线程
描述:
因为线程不属于spring托管，故线程不能够默认使用spring的事务,也不能获取spring注入的bean
在被spring声明式事务管理的方法内开启多线程，多线程内的方法不被事务控制
解决:
如果方法中调用多线程
方法主题的事务不会传递到线程中
线程中可以单独调用Service接口，接口的实现方法使用@Transactional，保证线程内部的事务
多线程实现的方法
使用异步注解@Async的方法上再加上注解@Transactional，保证新线程调用的方法是有事务管理的
原理:
Spring中事务信息存储在ThreadLocal变量中，变量是某个线程上进行的事务所特有的(这些变量对于其他线程中发生的事务来讲是不可见的，无关的)
单线程的情况下，一个事务会在层级式调用的Spring组件之间传播
在@Transactional注解的服务方法会产生一个新的线程的情况下，事务是不会从调用者线程传播到新建线程的



## 75.AOP编程

​		<!--// aop-->
​        <dependency>
​            <groupId>org.springframework.boot</groupId>
​            <artifactId>spring-boot-starter-aop</artifactId>
​        </dependency>

@Aspect 表明是一个切面类
@Component 将当前类注入到Spring容器内
@Pointcut 切入点，其中execution用于使用切面的连接点。
          使用方法：execution(方法修饰符(可选) 返回类型 方法名 参数 异常模式(可选)) ，可以使用通配符匹配字符，*可以匹配任意字符。
@Before 在方法前执行
@After 在方法后执行
@AfterReturning 在方法执行后返回一个结果后执行
@AfterThrowing 在方法执行过程中抛出异常的时候执行
@Around 环绕通知，就是可以在执行前后都使用，这个方法参数必须为ProceedingJoinPoint，proceed()方法就是被切面的方法，
上面四个方法可以使用JoinPoint，JoinPoint包含了类名，被切面的方法名，参数等信息。（可以获取方法的参数和值）	
@Around 可以修改方法的参数和方法的返回值， 	将修改的值传入到.proceed()方法中，然后如果想要修改返回的值，直接创建新的对象，返回回去即可。
@Around("point(name)")
public Object Around(ProceedingJoinPoint pjp) throws Throwable {      
    Object object = pjp.proceed(new Object[]{"新参数"});
    User user1 = new User();
    user1.setUsername("有趣");
    return user1;  //返回新的返回值，类型与方法原来的返回值相同
}

Pointcut 指示器
切点的表达式以 指示器 开始， 指示器 就是一种关键字，用来告诉 Spring AOP 如何匹配连接点，Spring AOP 提供了以下几种指示器
execution
within
this 和 target
args 和 @args
@target
@annotation
execution
该指示器用来匹配方法执行连接点，即匹配哪个方法执行，如
	@Pointcut("execution(public String aaric.springaopdemo.UserDao.findById(Long))")
上面这个切点会匹配在 UserDao 类中 findById 方法的调用，并且需要该方法是 public 的，返回值类型为 String，只有一个 Long 的参数。
切点的表达式同时还支持宽字符匹配，如
	@Pointcut("execution(* aaric.springaopdemo.UserDao.*(..))")
上面的表达式中，第一个宽字符 * 匹配 任何返回类型，第二个宽字符 * 匹配 任何方法名，最后的参数 (..) 表达式匹配 任意数量任意类型 的参数，也就是说该切点会匹配类中所有方法的调用。
within
如果要匹配一个类中所有方法的调用，便可以使用 within 指示器
	@Pointcut("within(aaric.springaopdemo.UserDao)")
这样便可以匹配该类中所有方法的调用了。同时，我们还可以匹配某个包下面的所有类的所有方法调用，如下面的例子
	@Pointcut("within(aaric.springaopdemo..*)")
@target
该指示器用于匹配连接点所在的类是否拥有指定类型的注解，如
	@Pointcut("@target(org.springframework.stereotype.Repository
target匹配目标对象的类型，即被代理对象的类型，例如A继承了B接口，则使用target（"B"），target（"A"）均可以匹配到A.
@this
如果目标对象没有实现任何接口，Spring AOP 会创建基于JDK的动态代理，这时候需要使用 this 指示器.
@annotation
该指示器用于匹配连接点的方法是否有某个注解
	@Pointcut("@annotation(org.springframework.scheduling.annotation.Async)")
@args
该函数接收一个注解类的类名，当方法的运行时入参对象标注了指定的注解时，匹配切点。
args
该函数接收一个类名，表示目标类方法入参对象是指定类（包含子类）时，切点匹配。

AOP应用场景
场景一： 记录日志
场景二： 监控方法运行时间 （监控性能）
场景三： 权限控制
场景四： 缓存优化 （第一次调用查询数据库，将查询结果放入内存对象， 第二次调用， 直接从内存对象返回，不需要查询数据库 ）
场景五： 事务管理 （调用方法前开启事务， 调用方法后提交关闭事务 ）

Spring实现AOP采用cglib和jdk动态代理两种方式，@EnableAspectJAutoProxy(proxyTargetClass=true)可以加开关控制，
如果不加，目标对象如果有实现接口，则使用jdk动态代理，如果没有就采用cglib（因为我们知道cglib是基于继承的））
@EnableAspectJAutoProxy：
表示开启AOP代理自动配置，如果配@EnableAspectJAutoProxy表示使用cglib进行代理对象的生成；
设置@EnableAspectJAutoProxy(exposeProxy=true)表示通过aop框架暴露该代理对象，aopContext能够访问.

CGLIB（Code Generation Library)是一个基于ASM的字节码生成库，它允许我们在运行时对字节码进行修改和动态生成。CGLIB通过继承方式实现代理。  
cglib需要引入cglib的jar包，spring-core的jar包，则无需引入，因为spring中包含了cglib。
cglib代理无需实现接口，通过生成类字节码实现代理，比反射稍快，不存在性能问题，但cglib会继承目标对象，需要重写方法，所以目标对象不能为final类。

jdk的动态代理，其只能代理接口
调用了Proxy.newProxyInstance(ClassLoader loader,Class<?>[] interfaces,InvocationHandler h) 静态方法。
通过该方法生成字节码，动态的创建了一个代理类，
interfaces参数是该动态类所继承的所有接口，而继承InvocationHandler 接口的类则是实现在调用代理接口方法前后的具体逻辑。
通过跟踪提示代码可以看出：当代理对象调用真实对象的方法时，其会自动的跳转到代理对象关联的handler对象的invoke方法来进行调用。

asm是assembly的缩写，是汇编的称号，对于java而言，asm就是字节码级别的编程。  
而这里说到的asm是指objectweb asm,一种.class的代码生成器的开源项目.  
ASM是一套java字节码生成架构，它可以动态生成二进制格式的stub类或其它代理类，  
或者在类被java虚拟机装入内存之前，动态修改类。
它被用于以下项目：
openjdk，实现lambda表达式调用， Nashorn编译器
Groovy和Kotlin编译器
Cobertura 和Jacoco，测量代码范围
CGLIB动态代理类





## 76、pom optional true

pom <optional>true</optional> 防止将依赖传递到其他模块中造成冲突

<dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <optional>true</optional> <!-- 防止将devtools依赖传递到其他模块中 -->

</dependency>





## 77、springboot idea热部署

1)添加环境变量：
<dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-devtools</artifactId>
        <optional>true</optional> <!-- 防止将devtools依赖传递到其他模块中 -->
</dependency>
2)settings->compiler勾选build project automatically
3)alt + shift + ctrl + / 选择registry... 勾选：compiler.automake.allow.when.app.running
4)修改代码，切换到其它应用idea就会重新编译

spring.devtools.restart.exlude=config/** #排除某些内容