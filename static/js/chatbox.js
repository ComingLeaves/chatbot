var send =document.getElementById('send');
var pic =document.getElementById('pic');
var txt =document.getElementById('inp');
var info_box = document.getElementsByClassName('info_box')[0];

var onoff=true;
/*
pic.onclick=function(){
	if(onoff){
		pic.src='img/1.jpg';
		onoff=false;
	}
	else{
		pic.src='img/2.jpg';
		onoff=true;
	}
};
*/

send.onclick = function(){
	if(txt.value==''){
		alert('请输入内容');
	}
	else{

		var nDiv = document.createElement('div');
		var spans = document.createElement('span');
		var imgs = document.createElement('img');
		var sTxt = document.createTextNode(txt.value);
		var info_box = document.getElementsByClassName('info_box')[0];
		spans.appendChild(sTxt);
		nDiv.appendChild(spans);
		nDiv.appendChild(imgs);
		// nDiv.style.display='block';
		info_box.insertBefore(nDiv,info_box.lastChild);
		spans.className='infor';
	    nDiv.className='info_r';
	    imgs.src='/static/img/2.jpg';

		//ajax交互
		var qstr = txt.value;
		var astr = null;
		$.ajax({
            type: "POST",
            url: '/testPost',
            data: { 'qstr':qstr },
            dataType: 'json',
            success: function(data){
                astr = data["astr"];
                var nDiv = document.createElement('div');
                var spans = document.createElement('span');
                var imgs = document.createElement('img');
                var sTxt = document.createTextNode(astr);
                var info_box = document.getElementsByClassName('info_box')[0];
                spans.appendChild(sTxt);
                nDiv.appendChild(spans);
                nDiv.appendChild(imgs);
                // nDiv.style.display='block';
                info_box.insertBefore(nDiv,info_box.lastChild);
                spans.className='infol';
                nDiv.className='info_l';
                imgs.src='/static/img/1.jpg';
            },
            error: function(e){
                alert("error");
            }
        })

/*
		var nDiv = document.createElement('div');
		var spans = document.createElement('span');
		var imgs = document.createElement('img');
		var sTxt = document.createTextNode(astr);  //"你说啥？"
		var info_box = document.getElementsByClassName('info_box')[0];
		spans.appendChild(sTxt);
		nDiv.appendChild(spans);
		nDiv.appendChild(imgs);
		// nDiv.style.display='block';
		info_box.insertBefore(nDiv,info_box.lastChild);
	    spans.className='infol';
			nDiv.className='info_l';
			imgs.src='/static/img/1.jpg';

*/
	}
	txt.value='';
}


/*
var xmlhttp;
function loadXMLDoc(url)
{
xmlhttp=null;
if (window.XMLHttpRequest)
  {// all modern browsers
  xmlhttp=new XMLHttpRequest();
  }
else if (window.ActiveXObject)
  {// for IE5, IE6
  xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
  }
if (xmlhttp!=null)
  {
  xmlhttp.onreadystatechange=state_Change;
  xmlhttp.open("GET",url,true);
  xmlhttp.send(null);
  }
else
  {
  alert("Your browser does not support XMLHTTP.");
  }
}

function state_Change()
{
if (xmlhttp.readyState==4)
  {// 4 = "loaded"
  if (xmlhttp.status==200)
    {// 200 = "OK"
    document.getElementById('heihei').innerHTML="This file was last modified on: " + xmlhttp.getResponseHeader('Last-Modified');
    }
  else
    {
    alert("Problem retrieving data:" + xmlhttp.statusText);
    }
  }
}

*/