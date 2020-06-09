//校验重复密码正确性
function KeyUp(){
    var pw1 = $('#pw1').val();
    var pw2 = $('#pw2').val();
    if(pw1==pw2){
        $('#btn').removeAttr('disabled');
    }
    else{
        $('#btn').attr('disabled','disabled');
    }

}