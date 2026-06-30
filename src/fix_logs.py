import os
import sys
import argparse


def remove_broken_per_rule_lines(folder):
    """
    Remove lines starting with 'per-rule acc: {0:' from all .log files in folder.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"[Error] Not a directory: {folder}")
        return

    log_files = [f for f in os.listdir(folder) if f.endswith('.log')]
    log_files.sort()

    total_files = 0
    total_removed = 0

    for filename in log_files:
        path = os.path.join(folder, filename)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[Error] Cannot read {filename}: {e}")
            continue

        target_prefix = "per-rule acc: {0:"
        cleaned = []
        removed = 0
        for line in lines:
            if line.lstrip().startswith(target_prefix):
                removed += 1
            else:
                cleaned.append(line)

        if removed > 0:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned)
                total_files += 1
                total_removed += removed
                print(f"[Fixed] {filename}: removed {removed} broken line(s)")
            except Exception as e:
                print(f"[Error] Cannot write {filename}: {e}")

    print(f"\nDone. Processed {len(log_files)} log file(s), "
          f"fixed {total_files} file(s), removed {total_removed} broken line(s) total.")


def main():
    parser = argparse.ArgumentParser(
        description="Remove broken 'per-rule acc: {0: ...' lines from log files."
    )
    parser.add_argument(
        'folder',
        nargs='?',
        default='logs',
        help="Folder containing .log files (default: logs)"
    )
    args = parser.parse_args()
    remove_broken_per_rule_lines(args.folder)


if __name__ == '__main__':
    main()
