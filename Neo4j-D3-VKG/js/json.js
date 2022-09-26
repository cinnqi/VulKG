var arr = [{ uuid: '1', name: '张三' },
{ uuid: '2', name: '李四' },
{ uuid: '3', name: '王五' },
{ uuid: '2', name: '李四' },
{ uuid: '1', name: '张三' },
{ uuid: '1', name: '张三' }]
 

function RemoveArr(arr) {
    var arr1 = []
    for (var i = 0; i < arr.length; i++) {
        if (arr1.map(x => x.uuid).indexOf(arr[i].uuid) == -1) {
            arr1.push(arr[i])
        }
    }
    return arr1
}
//
 
function jsonUniq(arrjson) {
        let arr1 = [arrjson[0]];
        arrjson.forEach(function (item1, idx1) {
            let flag = false;
            arr1.forEach(function (item2, idx2) {
                if (item1.uuid == item2.uuid) {
                    flag = true;
                    return;
                }
            })
            if (!flag ) {
                arr1.push(item1)
            }
        })
        return arr1;
}
