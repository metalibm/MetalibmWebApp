
var itemContainers = [].slice.call(document.querySelectorAll('.board-column-content'));
var columnGrids = [];
var boardGrid;



// Define the column grids so we can drag those
// items around.
// itemContainers.forEach(function (container) {
var container = itemContainers[0];


  // Instantiate column grid.
  var grid = new Muuri(container, {
	items: '.board-item',
	layoutDuration: 400,
	layoutEasing: 'ease',
	dragEnabled: true,
	dragSort: function () {
	  return columnGrids;
	},
	dragSortInterval: 0,
	dragContainer: document.body,
	dragReleaseDuration: 400,
	dragReleaseEasing: 'ease'
  })
  .on('dragStart', function (item) {
	// Let's set fixed widht/height to the dragged item
	// so that it does not stretch unwillingly when
	// it's appended to the document body for the
	// duration of the drag.
	item.getElement().style.width = item.getWidth() + 'px';
	item.getElement().style.height = item.getHeight() + 'px';
  })
  .on('dragEnd', function (item) {
        console.log(item.getElement().id);
        var board_pos = document.getElementsByClassName("board-column")[0].getBoundingClientRect();
        console.log(board_pos);

        var mouse_x = window.event.clientX;
        var drag_start_x = item.getElement()._drag_start_x;
        console.log(drag_start_x, mouse_x);
        var delta = mouse_x - board_pos.x;
        console.log(delta);
        if (delta - board_pos.width > 200 || delta < -200) {
            grid.remove(item);
            update_registered_pass_list();
        }
  })
  .on('dragReleaseEnd', function (item) {
	// Let's remove the fixed width/height from the
	// dragged item now that it is back in a grid
	// column and can freely adjust to it's
	// surroundings.
	item.getElement().style.width = '';
	item.getElement().style.height = '';
	// Just in case, let's refresh the dimensions of all items
	// in case dragging the item caused some other items to
	// be different size.
	columnGrids.forEach(function (grid) {
	  grid.refreshItems();
    update_registered_pass_list();
	});
  })
  .on('layoutStart', function () {
	// Let's keep the board grid up to date with the
	// dimensions changes of column grids.
	boardGrid.refreshItems().layout();
  });

  // Add the column grid reference to the column grids
  // array, so we can access it later on.
  columnGrids.push(grid);

//});

// Instantiate the board grid so we can drag those
// columns around.
boardGrid = new Muuri('.board', {
  layoutDuration: 400,
  layoutEasing: 'ease',
  dragEnabled: true,
  dragSortInterval: 0,
  dragStartPredicate: {
	handle: '.board-column-header'
  },
  dragReleaseDuration: 400,
  dragReleaseEasing: 'ease'
});
function remove_pass(pass_id_tag) {
    var pass_elt = document.getElementById(pass_id_tag);
    grid.remove([pass_elt], {removeElements: true})
}
function get_item_list() {
	return grid.getItems().map(item => item.getElement().id.split('-')[1]);
}

function update_registered_pass_list()
{
    // update hidden pass list
    var new_pass_list = get_item_list();
    document.getElementById("registered_pass_list").setAttribute("value", new_pass_list);
    console.log("updating pass list to: ", new_pass_list);
}

/** Clean all passes already registered */
function clear_passes() {
    console.log("removing every registered passes");
    grid.remove(grid.getItems(), {removeElements: true});
}

/** Add a new pass selected using the "new_pass" menu */
function add_new_pass() {
    var pass_name = document.getElementById("new_pass").value;
    add_new_pass_by_name(pass_name);
}

/** add a new pass selected by the @p pass_name argument */
function add_new_pass_by_name(pass_name) {
    var item_id = "pass_" + pass_number.toString() + "-" + pass_name;
    pass_number++;
	var new_item = document.createElement("div");
	var new_item_content = document.createElement("div");
	new_item_content.innerHTML = pass_name;
	new_item_content.setAttribute("class", "board-item-content");

	new_item.appendChild(new_item_content);
	new_item.setAttribute("class", "board-item");
	new_item.setAttribute("id", item_id);
	grid.add(new_item);
	boardGrid.refreshItems().layout();

    update_registered_pass_list();
}

function add_pass_list(pass_list)
{
    for (var i = 0; i < pass_list.length; ++i) {
        add_new_pass_by_name(pass_list[i])
    };
}

/** When triggered, look at selected pre-configuration flow,
 *  clear all registered passes and add the passes corresponding
 *  to the selected flow
 */
function add_preconf_flow() {
    clear_passes();
	var preconf_selection = document.getElementById("preconf_flow");
    var selected_flow = preconf_selection.options[preconf_selection.selectedIndex].value;
    console.log("select flow: " + selected_flow);
    add_pass_list(pre_conf_map[selected_flow]);
}

function copyCodeToClipboard() {
	var copyText = document.getElementById("sourceCode");
	updateClipboard(copyText.value)
	//navigator.permissions.query({name: "clipboard-write"}).then(result => { 
	//  if (result.state == "granted" || result.state == "prompt") {
	//	/* write to the clipboard now */
	//	navigator.clipboard.writeText(copyText.value)
	//  }
	//});
}

/** Copying data to clipboard */
function updateClipboard(newClip) {
    navigator.clipboard.writeText(newClip).then(function() {
        /* clipboard successfully set */
    }, function() {
        /* clipboard write failed */
        alert("copy in clipboard failed");
    });
}
