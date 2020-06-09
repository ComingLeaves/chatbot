const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');
const container = document.getElementById('container');

signUpButton.addEventListener('click', () => {
	container.classList.add("right-panel-active");
});

signInButton.addEventListener('click', () => {
	container.classList.remove("right-panel-active");
});




/*
var email =document.getElementById('email');
var password =document.getElementById('password');
function login(){
    	if(email.value=='' && password.value==''){
		    alert("请输入注册邮箱和密码");
	    }
	    else{
	        $.ajax({
                type: "POST",
                url: '/loginPost',
                data: { 'email':email ,'password':password},
                dataType: 'json',
                success: function(data){
                    if(data['flag']==1){
                        alert("登陆成功");
                    }else{
                        alert("邮箱或密码错误");
                    }

                },
                error: function(e){
                    alert("error");
                }
            })
	    }

}
*/