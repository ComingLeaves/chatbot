var content = document.getElementsByClassName("content")[0],
    today = new Date();
//定义默认时间
var defaultDate = today,
    year = defaultDate.getFullYear(),
    month = defaultDate.getMonth(),
    dateDay = defaultDate.getDate;
var preObj,arrCalendar;
//节日对应的对象，后期可手动维护
var festival = {
    1: {1: "元旦"},
    2: {14: "情人节"},
    3: {8: "妇女节",15: "315"},
    4: {1: "愚人节"},
    5: {1: "劳动节",4: "青年节",12: "护士节"},
    6: {1: "儿童节"},
    7: {1: "建党节"},
    8: {1:"建军节"},
    9: {10: "教师节"},
    10: {1: "国庆节"},
    11: {11: "光棍节"},
    12: {24: "平安夜",25: "圣诞节"}
},
    festivalB = {
        2016: {
            1: {1: "元旦",17: "腊八节"},
            2: {1: "祭灶节",2: "除夕",8: "春节", 12: "破五", 14: "情人节",22: "元宵节"},
            3: {8: "妇女节",10: "二月二龙抬头", 12: "植树节", 15: "3·15消费者权益日",27: "复活节"},
            4: {1: "愚人节",4: "清明节",29: "妈祖生辰"},
            5: {1: "劳动节",4: "青年节",8: "母亲节",12: "护士节"},
            6: {1: "儿童节",9: "端午节", 19: "父亲节"},
            7: {1: "建党节",27: "火把节（彝族）"},
            8: {1: "建军节",9: "七夕"},
            9: {10: "教师节",15: "中秋节"},
            10: {1: "国庆节", 9: "重阳节", 31: "万圣节前夜"},
            11: {11: "光棍节",14: "下元节",24: "感恩节"},
            12: {1: "世界艾滋病日", 13: "南京大屠杀纪念日", 24: "平安夜",25: "圣诞节"}
        }},
    //假期对应的对象，后期可手动维护(A对应每年固定的部分，B对应手动添加的部分）
    vacA = {
        1: [1],
        5: [1],
        10: [1]
    },
    vacB = {2015: {1: [1,2,3],2: [18,19,20,21,22,23,24],4: [5,6],5: [1,2,3],6: [20,21,22],9: [27],10: [1,2,3,4,5,6,7]}
        ,2016: {1: [1,2,3], 2: [7,8,9,10,11,12,13],4: [2,3,4,30],5: [1,2],6: [9,10,11],9: [15,16,17],10: [1,2,3,4,5,6,7]}
    };
//补零函数
function addZero(s) {
    return s < 10 ? '0' + s: s;
}

//初始化
function initial() {
    if(content.childNodes.length > 0) {
        content.parentElement.removeChild(content);
        content = document.createElement("div");
        content.className = ("content");
        document.getElementsByClassName("outerContainer")[0].appendChild(content);
    }
}

//绘制Calendar
function drawCalendar(arrDate,i) {
    var nDiv,
        clsName = [" preMon"," theMon"," nxtMon"],
        nSpan,
        nInput;
    nDiv = document.createElement("div");
    nSpan = document.createElement("span");
    nInput = document.createElement("input");
    nInput.style = "display: none";
    nInput.value = arrDate;
    nSpan.className = "festival";
    nDiv.className = "date";
    nDiv.className += clsName[i];
    addFestival(arrDate,nSpan,nDiv);
    nDiv.innerHTML = arrDate.getDate();
    addClick(nDiv);
    nDiv.appendChild(nInput);
    nDiv.appendChild(nSpan);
    content.appendChild(nDiv);
}

//生成当前月日期对象的数组，绘制日历
function getArr(y,mon) {
    var arr = [],
        days = new Date(y,mon,0).getDate(), //确定本月有多少天
        daysPre = new Date(y,mon - 1,0).getDate(), // 确定上个月有多少天
        datDay = new Date(y,mon - 1,1).getDay(), //确定本月首日是周几
        arrDate,
        k = 1;
    if(datDay == 0) {
        datDay = 7
    }
    for(var i = daysPre - datDay;i < daysPre;i++) {
        arrDate = new Date(y,mon-2,i + 1); //生成上个月末尾的日期
        arr.push(arrDate);
        drawCalendar(arrDate,0);
    }
    for(var j = 0;j < days;j++) {
        arrDate = new Date(y,mon-1,j + 1); //生成这个月的日期
        arr.push(arrDate);
        drawCalendar(arrDate,1);
    }
    while(arr.length < 42) {
        arrDate = new Date(y,mon,k);
        arr.push(arrDate);
        drawCalendar(arrDate,2);
        k ++;
    }
    arrCalendar = arr;
}
//下面一段代码更加优雅的实现方法？？？

//确定当天是否是节日，及是否是假期
function addFestival(arrDate,spanObj,divObj) {
    if(festivalB[arrDate.getFullYear()]){
        if(festivalB[arrDate.getFullYear()][arrDate.getMonth() + 1]) {
            if(festivalB[arrDate.getFullYear()][arrDate.getMonth() + 1][arrDate.getDate()]){
                spanObj.innerHTML = festivalB[arrDate.getFullYear()][arrDate.getMonth() + 1][arrDate.getDate()];
            }
        }
    }else if(festival[arrDate.getMonth() + 1]){
        if(festival[arrDate.getMonth() + 1][arrDate.getDate()]) {
            spanObj.innerHTML = festival[arrDate.getMonth() + 1][arrDate.getDate()];
        }
    }
    if(vacB[arrDate.getFullYear()] && vacB[arrDate.getFullYear()][arrDate.getMonth() + 1]) {
        if(vacB[arrDate.getFullYear()][arrDate.getMonth() + 1].indexOf(arrDate.getDate()) != -1){
            divObj.className += " vacation";
        }
    }else if(vacA[arrDate.getMonth() + 1]) {
        if(vacA[arrDate.getMonth() + 1].indexOf(arrDate.getDate()) != -1){
            divObj.className += " vacation";
        }
    }
}

//设置默认时间的样式
function setDefault() {
    for(b in arrCalendar){
        if(defaultDate - arrCalendar[b] < 86400000 && defaultDate - arrCalendar[b] > 0){
            var divs = document.getElementsByClassName("date");
            preObj = divs[b];
            displayDate(preObj);
            preObj.className += " click";
        }
    }
}

//给按钮添加点击的事件
function addClick(obj) {
    obj.addEventListener("click",function() {
        if(preObj) {
            if(preObj.className.indexOf(" click") != -1) {
                preObj.className = preObj.className.replace(" click","");
            }
        }
        obj.className += " click";
        displayDate(obj);
        preObj =  obj;
    })
}

//计算距现在的时间
function daysFromNow(dateObj,dateElem) {
    var daysFromNow = (dateObj - today) / 86400000;
    if(daysFromNow > 0) {
        daysFromNow = ~~daysFromNow + 1;
    }if(daysFromNow <= 0) {
        daysFromNow = ~~daysFromNow;
    }
    if(daysFromNow > 0) {
        dateElem.innerHTML = daysFromNow + "天之后";
    }else if(daysFromNow < 0) {
        dateElem.innerHTML = -daysFromNow + "天之前";
    }else{
        dateElem.innerHTML = "今天"
    }
}

//显示日期
function displayDate(obj) {
    var dateT = document.getElementsByClassName("dateT")[0],
        yearT = document.getElementsByClassName("yearT")[0],
        daysFrom = document.getElementsByClassName("daysFrom")[0],
        inputContent = obj.getElementsByTagName("input")[0].value,
        inputDate = new Date(inputContent);
    dateT.innerHTML = inputDate.getMonth() + 1 + "月" + inputDate.getDate() + "日";
    yearT.innerHTML = inputDate.getFullYear();
    daysFromNow(inputDate,daysFrom);
}

//显示时间
function displayTime() {
    var nowD  = new Date(),
        hours = nowD.getHours(),
        minutes = nowD.getMinutes(),
        seconds = nowD.getSeconds(),
        right = document.getElementsByClassName("right")[0];
    if(hours > 12) {
        hours = hours - 12;
        right.innerHTML = "下午" + hours + ":" + addZero(minutes) + ":" + addZero(seconds);
    }else{
        right.innerHTML = "上午" + hours + ":" + addZero(minutes) + ":" + addZero(seconds);
    }
}

//添加键盘事件
function addKey() {
    document.addEventListener("keydown",function(event){
        var e = event || window.event || arguments.callee.caller.arguments[0];
        function reDraw(){
            initial();
            getArr(year,month + 1);
        }
        switch(e.keyCode){
            case 37:
                month --;
                reDraw();
                break;
            case 40:
                year ++;
                reDraw();
                break;
            case 39:
                month ++;
                reDraw();
                break;
            case 38:
                year --;
                reDraw();
                break;
        }
    })
}
initial();
getArr(year,month + 1);
setInterval(displayTime, 100);
setDefault();
addKey();