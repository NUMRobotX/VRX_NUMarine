savedcmd_/home/nuc1/Downloads/linuxcan/pciefd/kvpciefd.mod := printf '%s\n'   pciefd_hwif.o hw/pciefd_altera.o hw/pciefd_sf2.o hw/pciefd_xilinx.o drivers/kvaser/pwm/pwm_util.o drivers/kvaser/pciefd/pciefd.o drivers/kvaser/pciefd/pciefd_packet.o drivers/kvaser/pciefd/pciefd_rx_fifo.o drivers/altera/HAL/src/altera_avalon_epcs_flash_controller.o drivers/altera/HAL/src/altera_avalon_spi.o drivers/altera/HAL/src/epcs_commands.o drivers/kvaser/spi/sf2_spi/sf2_spi.o drivers/xilinx/spi/src/xspi.o drivers/xilinx/spi/src/xspi_options.o drivers/kvaser/spi_flash/spi_flash.o drivers/kvaser/spi_flash/spi_flash_altera.o drivers/kvaser/spi_flash/spi_flash_sf2.o drivers/kvaser/spi_flash/spi_flash_xilinx.o drivers/kvaser/spi_flash/spi_flash_dummy.o drivers/kvaser/hydra_flash/hydra_flash.o | awk '!x[$$0]++ { print("/home/nuc1/Downloads/linuxcan/pciefd/"$$0) }' > /home/nuc1/Downloads/linuxcan/pciefd/kvpciefd.mod