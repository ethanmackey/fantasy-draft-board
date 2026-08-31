/**
 * Draft Board -- one-click "player is gone", one-click "start over".
 *
 * Paste into the spreadsheet's Extensions > Apps Script editor and save. No
 * triggers to install and no authorisation prompt: onEdit and onOpen are simple
 * triggers, which may edit their own spreadsheet.
 *
 * The board's geometry is baked in below, written by the generator out of the same
 * constants it laid the board out with, so the two cannot drift. layout() is still
 * here and still the authority: it finds the header row, treats every bare "Rank"
 * header as a position block -- checkbox one column left, player name one column
 * right -- and locates the log by its "Drafted" header. Anything the baked answer
 * gets wrong falls through to it, so rebuilding the board with different columns,
 * widths or row offsets still does NOT require pasting this again.
 *
 * Ticking the checkbox beside a player writes their name into the hidden drafted
 * log and immediately unticks the box. Each position block is a FILTER over a
 * hidden source that excludes every name in that log, so the player vanishes from
 * their own column and everyone below slides up -- and because the box is already
 * clear, whoever slides into that row is not swept up with them.
 *
 * Ticking A1 resets the board. It copies the log to a backup column before
 * clearing, so a stray click is recoverable from the Draft Board menu.
 */

var SHEET_NAME = 'Draft Board';
var RESET_ROW = 1, RESET_COL = 1;   // the reset box in A1
var SEARCH_ROWS = 8;                // the header row is within the first few

/**
 * Where everything is, written in at build time.
 *
 * Working it out at runtime instead is what layout() does, and it costs two round
 * trips to the spreadsheet -- getLastColumn(), then an eight-row probe across the
 * full width of the sheet -- before the script has done anything at all. That is
 * the wrong price on the one path whose whole point is that a tick feels instant,
 * and it was paid on every pick.
 *
 * Nothing trusts this blindly. The guards below fall back to layout() when the
 * baked columns do not match the edit, and the read that fetches the player's name
 * checks that it landed on a column headed "Player" before drafting anybody.
 */
var BAKED = {
  headerRow: 4,          // 1-indexed, the way the sheet counts
  firstDataRow: 5,
  checkboxCols: [1, 11, 21, 31],   // one per position block
  nameOffset: 2,             // checkbox | Rank | Player
  toggleRow: 2, toggleCol: 10,   // the TE premium box
  drafted: 85,
  backup: 86,
  logLastRow: 220          // DRAFTED_LIMIT
};


/**
 * Work out where everything is by reading the header row.
 * Returns {headerRow, firstDataRow, checkboxCols, nameOffset, drafted, backup}
 * or null.
 */
function layout(sheet) {
  var probe = sheet.getRange(1, 1, SEARCH_ROWS, sheet.getLastColumn()).getValues();

  var headerRow = -1;
  for (var r = 0; r < probe.length && headerRow === -1; r++) {
    for (var c = 0; c < probe[r].length; c++) {
      if (probe[r][c] === 'Player') { headerRow = r; break; }
    }
  }
  if (headerRow === -1) return null;

  var head = probe[headerRow];
  var checkboxCols = [], drafted = -1, backup = -1;
  for (var i = 0; i < head.length; i++) {
    // A bare "Rank" heads a visible block. The hidden source columns are named
    // "QB Rank", "RB Rank" and so on, so they never match here.
    if (head[i] === 'Rank' && i > 0) checkboxCols.push(i);  // 1-indexed checkbox
    if (head[i] === 'Drafted') drafted = i + 1;
    if (head[i] === 'Drafted (backup)') backup = i + 1;
  }
  if (!checkboxCols.length || drafted === -1) return null;

  return {
    headerRow: headerRow + 1,            // 1-indexed, the way the sheet counts rows
    firstDataRow: headerRow + 2,
    checkboxCols: checkboxCols,
    nameOffset: 2,                       // checkbox | Rank | Player
    drafted: drafted,
    backup: backup === -1 ? drafted + 1 : backup
    // No logLastRow: nothing on the sheet says where the log ends, so logRange()
    // falls back to getLastRow() for a geometry that came from here.
  };
}


function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Draft Board')
    .addItem('Undo last pick', 'undoLastPick')
    .addItem('Undo reset (restore cleared board)', 'undoReset')
    .addSeparator()
    .addItem('Reset board (put everyone back)', 'resetBoard')
    .addToUi();
}


function onEdit(e) {
  if (!e || !e.range) return;
  if (e.value !== 'TRUE') return;              // only care about a box being ticked

  var range = e.range;
  var row = range.getRow();
  var col = range.getColumn();
  var sheet = range.getSheet();
  if (sheet.getName() !== SHEET_NAME) return;

  if (row === RESET_ROW && col === RESET_COL) {
    // Not a hot path -- it asks the sheet where everything is, and if the answer
    // is unreadable the baked geometry beats doing nothing.
    var reset = layout(sheet) || BAKED;
    range.setValue(false);                     // clear the box, then wipe the board
    clearBoard(sheet, reset, true);
    SpreadsheetApp.getActive().toast(
      'Board reset. Draft Board > Undo reset puts the picks back.');
    return;
  }

  // The TE premium toggle is the board's other checkbox and it is not a pick. It
  // is named explicitly so that it returns here for nothing, instead of falling
  // into the re-read below every time somebody switches scoring.
  if (row === BAKED.toggleRow && col === BAKED.toggleCol) return;

  // The guards themselves run off the baked geometry, before any read.
  //
  // The re-read is for a board that has been rebuilt with different columns, which
  // is the only thing that makes the baked answer wrong -- so it is on the path
  // that would otherwise silently do nothing on a click, and off the path of every
  // ordinary edit. Everything else that is not a pick has already returned: a name
  // typed into the log or a target list is not the string TRUE, and the reset box
  // and the toggle are both handled above.
  var L = BAKED;
  if (row < L.firstDataRow || L.checkboxCols.indexOf(col) === -1) {
    L = layout(sheet);                         // the baked answer does not fit
    if (!L) return;
    if (row < L.firstDataRow || L.checkboxCols.indexOf(col) === -1) return;
  }

  // BOTH reads before EITHER write, and that ordering is the whole difference
  // between one recalculation of the board per pick and two.
  //
  // Reads and writes do not interleave for free: a read flushes whatever writes
  // are queued, and the flush recalculates every FILTER, every value column and
  // every conditional format on the board. Unticking the box and THEN reading the
  // log -- which is what this used to do -- forced that recalculation in the
  // middle of the script, and the writes at the end of it forced a second. Reads
  // first, and the untick and the logged name flush together, once.
  var name = readName(sheet, L, row, col);
  if (name === null) {                         // not a name column after all
    L = layout(sheet);
    if (!L || row < L.firstDataRow || L.checkboxCols.indexOf(col) === -1) return;
    name = readName(sheet, L, row, col);
    if (name === null) return;
  }
  var logged = name ? logRange(sheet, L, L.drafted).getValues() : null;

  // Clear the box first. If the name came back empty -- a click on a blank row
  // past the end of a position -- that is all that happens.
  range.setValue(false);
  if (!name) return;

  // No toast, deliberately. It was worth one round trip while the wait was being
  // measured -- 400-800ms from click to toast, with the row collapsing straight
  // after it, which put the whole cost in Google's trigger dispatch and this
  // script rather than in the board's formulas. It is not worth one per pick to
  // repeat something the vanishing row already says.
  markDrafted(sheet, L, name, logged);
}


/**
 * The player beside a ticked checkbox, in a read that also proves the column.
 *
 * The range runs from the header row down to the player's own cell, so the first
 * value it returns is that column's heading and the last is the name. A block's
 * name column is headed "Player"; anything else means the geometry is pointing
 * somewhere it should not be, and null says so rather than drafting whatever
 * happened to be sitting there. It costs no more than reading the one cell: a
 * couple of hundred values down a single column is one round trip either way.
 */
function readName(sheet, L, row, col) {
  var span = sheet.getRange(L.headerRow, col + L.nameOffset,
                            row - L.headerRow + 1, 1).getValues();
  if (span[0][0] !== 'Player') return null;
  return span[span.length - 1][0];
}


/**
 * The drafted log, or its backup. Its last row is a build-time constant, so the
 * baked geometry knows it and getLastRow() -- one more round trip on the hot path
 * -- is only needed for a geometry that came off the sheet.
 */
function logRange(sheet, L, column) {
  var lastRow = L.logLastRow || Math.max(sheet.getLastRow(), L.firstDataRow);
  return sheet.getRange(L.firstDataRow, column, lastRow - L.firstDataRow + 1, 1);
}


/**
 * A whole column of the board proper, which runs PAST the end of the log: there
 * are more players than a draft has picks, so the checkboxes go further down than
 * the log does and a sweep over them cannot use logRange's bound.
 */
function boardRange(sheet, L, column) {
  var lastRow = Math.max(sheet.getLastRow(), L.firstDataRow);
  return sheet.getRange(L.firstDataRow, column, lastRow - L.firstDataRow + 1, 1);
}


/**
 * Append a name to the drafted log, if it is not already there.
 *
 * The log's values come in from onEdit, which read them before it wrote anything
 * -- see the note there on why the order matters. Read here instead when there is
 * nothing to pass in, which is every caller except the checkbox.
 */
function markDrafted(sheet, L, name, logged) {
  if (!logged) logged = logRange(sheet, L, L.drafted).getValues();

  var firstEmpty = -1;
  for (var i = 0; i < logged.length; i++) {
    if (logged[i][0] === name) return;         // already off the board
    if (firstEmpty === -1 && logged[i][0] === '') firstEmpty = i;
  }
  var target = firstEmpty === -1 ? logged.length : firstEmpty;
  sheet.getRange(L.firstDataRow + target, L.drafted).setValue(name);
}


/** Empty the drafted log and untick every box. Optionally back the log up first. */
function clearBoard(sheet, L, backup) {
  var log = logRange(sheet, L, L.drafted);
  if (backup) {
    var values = log.getValues();
    logRange(sheet, L, L.backup).clearContent();
    sheet.getRange(L.firstDataRow, L.backup, values.length, 1).setValues(values);
  }
  log.clearContent();

  // Clear any box left ticked -- only possible if a paste bypassed onEdit.
  //
  // Only where a checkbox actually exists. setValue(false) down the whole column
  // wrote a literal FALSE into every row past the end of the position -- ~190
  // under quarterback, ~155 under tight end -- so the one control that is always
  // on screen left the board covered in the word FALSE. Clearing the cells that
  // hold no checkbox, and unticking only the ones that do, leaves it blank.
  for (var i = 0; i < L.checkboxCols.length; i++) {
    var column = boardRange(sheet, L, L.checkboxCols[i]);
    var rules = column.getDataValidations();
    var out = [];
    for (var r = 0; r < rules.length; r++) {
      var rule = rules[r][0];
      var isBox = rule && rule.getCriteriaType() ===
                  SpreadsheetApp.DataValidationCriteria.CHECKBOX;
      out.push([isBox ? false : '']);
    }
    column.setValues(out);
  }
}


function active() {
  var sheet = SpreadsheetApp.getActive().getSheetByName(SHEET_NAME);
  return {sheet: sheet, L: layout(sheet) || BAKED};
}


function undoLastPick() {
  var a = active();
  var values = logRange(a.sheet, a.L, a.L.drafted).getValues();
  for (var i = values.length - 1; i >= 0; i--) {
    if (values[i][0] !== '') {
      a.sheet.getRange(a.L.firstDataRow + i, a.L.drafted).clearContent();
      SpreadsheetApp.getActive().toast(values[i][0] + ' is back on the board');
      return;
    }
  }
  SpreadsheetApp.getActive().toast('No picks to undo');
}


function undoReset() {
  var a = active();
  var backup = logRange(a.sheet, a.L, a.L.backup).getValues();
  var picks = backup.filter(function (row) { return row[0] !== ''; }).length;
  if (!picks) {
    SpreadsheetApp.getActive().toast('Nothing to restore');
    return;
  }
  var log = logRange(a.sheet, a.L, a.L.drafted);
  log.clearContent();
  a.sheet.getRange(a.L.firstDataRow, a.L.drafted, backup.length, 1).setValues(backup);
  SpreadsheetApp.getActive().toast('Restored ' + picks + ' pick(s)');
}


function resetBoard() {
  var ui = SpreadsheetApp.getUi();
  var answer = ui.alert('Reset the board?',
                        'This clears every pick and puts all players back.',
                        ui.ButtonSet.YES_NO);
  if (answer !== ui.Button.YES) return;

  var a = active();
  clearBoard(a.sheet, a.L, true);
  SpreadsheetApp.getActive().toast('Board reset');
}
